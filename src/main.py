import argparse, sys, os
from render import Render

def create_arg_parser():
    arg_parser = argparse.ArgumentParser(
            description='Renders a 2d isometric city from a tilemap.'
            )
    arg_parser.add_argument(
            'tilemap',
            type=str,
            help='Relative path to tilemap'
            )
    return arg_parser

def loadTilemap(filepath):
    return [list(line[:-1]) for line in open(filepath,'r')]


if __name__ == "__main__":
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    tilemap = []
    if os.path.exists(parsed_args.tilemap):
        tilemap = loadTilemap(parsed_args.tilemap)

    print(tilemap)

    r = Render(tilemap)
    r.render()
    while True:
        pass
