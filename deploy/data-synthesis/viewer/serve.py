"""Serve the existing three-camera annotation UI on loopback, using CPU only."""
import argparse
from pathlib import Path
from app import create_app
from build_index import build_index, freeze_index
from waitress import create_server


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('/source'))
    p.add_argument('--review', type=Path, default=Path('/review'))
    p.add_argument('--workers', nargs='+', type=int, required=True)
    p.add_argument('--export-name', default='export_v1')
    p.add_argument('--port', type=int, default=18765)
    args = p.parse_args()
    if not 1024 <= args.port <= 65535:
        p.error('Choose a port between 1024 and 65535')
    if args.review.resolve().is_relative_to(args.source.resolve()):
        p.error('Use a separate review directory outside --source')
    print('Indexing exported trajectories...', flush=True)
    rows = build_index(args.source, args.workers, args.export_name)
    index = freeze_index(args.review/'episodes.json', rows)
    app = create_app(args.source, index, args.review/'labels')
    server = create_server(app, host='127.0.0.1', port=args.port, threads=4)
    print(f'VIEWER_READY: {len(rows)} episodes; http://127.0.0.1:{args.port}', flush=True)
    server.run()


if __name__ == '__main__':
    main()
