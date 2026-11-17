import argparse
import json
from pathlib import Path

from multi_agent.memory import Memory
from multi_agent.workflow import HERE


def main():
    parser = argparse.ArgumentParser(description='Помощник по фильмам MovieLens 100K')
    parser.add_argument('--directory', type=Path, default=HERE / 'var')
    sub = parser.add_subparsers(dest='command', required=True)
    ask = sub.add_parser('ask')
    ask.add_argument('request')
    ask.add_argument('--user', type=int, default=1)
    ask.add_argument('--output', type=Path)
    profile = sub.add_parser('profile')
    profile.add_argument('--user', type=int, required=True)
    profile.add_argument('--prefer', nargs='*', default=[])
    profile.add_argument('--avoid', nargs='*', default=[])
    forget = sub.add_parser('forget')
    forget.add_argument('--user', type=int, required=True)
    args = parser.parse_args()
    if args.command == 'ask':
        from multi_agent.llm import ModelRouter
        from multi_agent.workflow import MovieAssistant
        app = MovieAssistant(ModelRouter(work_dir=args.directory / 'engine'), args.directory)
        try:
            result = app.ask(args.request, args.user)
        finally:
            app.close()
        text = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        print(text)
    else:
        memory = Memory(args.directory / 'memory.sqlite3')
        try:
            if args.command == 'profile':
                print(json.dumps(memory.update_profile(args.user, args.prefer, args.avoid),
                                 ensure_ascii=False, indent=2))
            else:
                memory.forget(args.user)
                print('Профиль и сводки запусков удалены')
        finally:
            memory.close()


if __name__ == '__main__':
    main()
