"""Изоляция учебного запуска на macOS Seatbelt. Загрузка весов выполняется отдельно."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def policy(directory):
    config = json.loads((ROOT / 'multi_agent/models.json').read_text())
    port = int(config['inference']['port'])
    paths = [ROOT / p for p in ('multi_agent', 'recsys', '.venv', '.cache/engine', '.cache/models')]
    paths.append(directory.resolve())
    read_rules = '\n'.join(f'(subpath {json.dumps(str(p))})' for p in paths)
    # dyld/Python читают сами каталоги cwd и его предков; это не даёт читать их файлы.
    ancestors = '\n'.join(f'(literal {json.dumps(str(p))})' for p in [ROOT, *ROOT.parents])
    engine = ROOT / config['engine']['path']
    return f'''(version 1)
(deny default)
(allow file-read*
  (subpath "/System") (subpath "/usr") (subpath "/bin")
  (subpath "/Library/Frameworks/Python.framework")
  (subpath "/private/var/db/dyld") (subpath "/dev")
  {read_rules}
  {ancestors})
(allow file-read-metadata)
(allow file-write* (subpath {json.dumps(str(directory.resolve()))}) (literal "/dev/null"))
(allow process-fork)
(allow process-exec (subpath "/Library/Frameworks/Python.framework")
  (literal {json.dumps(str(engine))}))
(allow signal (target same-sandbox))
(allow sysctl-read)
(allow mach-lookup)
(allow network-outbound (remote ip "localhost:{port}"))
(allow network-inbound (local ip "localhost:{port}"))
(allow network-bind (local ip "localhost:{port}"))
'''


def launch(directory, arguments):
    if sys.platform != 'darwin' or not Path('/usr/bin/sandbox-exec').exists():
        raise RuntimeError('Этот launcher требует macOS sandbox-exec; не запускаю без изоляции')
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'agent.sb'
    path.write_text(policy(directory))
    env = {k: v for k, v in os.environ.items() if k in ('PATH', 'LANG', 'LC_ALL', 'TMPDIR')}
    env.update({'PYTHONDONTWRITEBYTECODE': '1', 'LANGCHAIN_TRACING_V2': 'false',
                'LANGSMITH_TRACING': 'false', 'TOKENIZERS_PARALLELISM': 'false'})
    command = ['/usr/bin/sandbox-exec', '-f', str(path), str(ROOT / '.venv/bin/python'), *arguments]
    return subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)


def check(directory, output):
    canary = ROOT / 'verification/sandbox-canary.txt'
    canary.write_text('Учебный canary для проверки запрета чтения. Не секрет.\n')
    config = json.loads((ROOT / 'multi_agent/models.json').read_text())
    port = config['inference']['port']
    source = '''import json, socket, subprocess, sys
from pathlib import Path
checks=[]
def denied(name, operation):
    try: operation()
    except PermissionError: checks.append({'name':name,'expected':'denied','passed':True})
    except OSError as e: checks.append({'name':name,'expected':'denied','passed':e.errno in (1,13),'error':str(e)})
    else: checks.append({'name':name,'expected':'denied','passed':False})
denied('read_outside_allowlist', lambda: Path(CANARY).read_text())
denied('write_outside_runtime', lambda: Path(DENIED_WRITE).write_text('blocked'))
denied('external_network', lambda: socket.create_connection(('1.1.1.1',443),timeout=2))
try:
    Path(ALLOWED_WRITE).write_text('allowed')
    checks.append({'name':'write_runtime','expected':'allowed','passed':True})
except OSError as e: checks.append({'name':'write_runtime','passed':False,'error':str(e)})
try:
    server=socket.socket(); server.bind(('127.0.0.1', PORT)); server.listen(1)
    client=socket.create_connection(('127.0.0.1',PORT),timeout=2)
    conn,_=server.accept(); client.close(); conn.close(); server.close()
    checks.append({'name':'loopback_llm_port','expected':'allowed','passed':True})
except OSError as e: checks.append({'name':'loopback_llm_port','passed':False,'error':str(e)})
child=None
try:
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(5)'])
    child.terminate(); child.wait(timeout=2)
    checks.append({'name':'stop_own_llm_child','expected':'allowed','passed':True})
except (OSError, subprocess.TimeoutExpired) as e:
    checks.append({'name':'stop_own_llm_child','passed':False,'error':str(e)})
    if child is not None: child.wait(timeout=6)
print(json.dumps(checks))
'''
    parameters = {'CANARY': str(canary), 'DENIED_WRITE': str(ROOT / 'verification/denied-write.txt'),
                  'ALLOWED_WRITE': str(directory.resolve() / 'allowed-write.txt'), 'PORT': port}
    code = '\n'.join(f'{k}={v!r}' for k, v in parameters.items()) + '\n' + source
    completed = launch(directory, ['-c', code])
    try:
        checks = json.loads(completed.stdout)
    except json.JSONDecodeError:
        checks = [{'name': 'sandbox_start', 'passed': False, 'error': completed.stderr[-2000:]}]
    result = {'sandbox': 'macOS Seatbelt sandbox-exec', 'returncode': completed.returncode,
              'checks': checks, 'all_passed': completed.returncode == 0 and all(c['passed'] for c in checks),
              'policy': str(directory / 'agent.sb'),
              'limitation': 'Deprecated macOS facility; no VM-level kernel isolation or hard RAM/CPU quotas'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'multi_agent/var/sandbox')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'multi_agent/results/isolation.json')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.check:
        result = check(args.directory, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result['all_passed']:
            raise SystemExit(1)
    else:
        arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
        if not arguments:
            parser.error('После -- укажите Python arguments, например -m multi_agent ...')
        result = launch(args.directory, arguments)
        print(result.stdout, end='')
        print(result.stderr, end='', file=sys.stderr)
        sys.exit(result.returncode)


if __name__ == '__main__':
    main()
