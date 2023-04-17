import util.main as idscn
import config.parse as parse

if __name__ == '__main__':
    parser = parse.parse()
    args = parser.parse_args()
    if args.mode == 'gen':
        filepath = '/'.join(args.input.split('\\'))
        outpath = '/'.join(args.output.split('\\'))
        name_path = '/'.join(args.name.split('\\'))
        tp = args.type
        group_index = args.group_index
        group_name, cova_name, region_name = parse.parse_name(name_path, tp)
        idscn.generate_dataset(filepath, outpath, group_name, group_index, cova_name, region_name, tp)
    elif args.mode == 'IDSCN':
        inpath = '/'.join(args.input.split('\\'))
        outpath = '/'.join(args.output.split('\\'))
        name_path = '/'.join(args.name.split('\\'))
        _, cova_name, region_name = parse.parse_name(name_path)
        idscn.IDSCN(inpath=inpath, outpath=outpath, cova=cova_name, region=region_name)
    elif args.mode == 'cluster':
        input_dir = '/'.join(args.input.split('\\'))
        outpath = '/'.join(args.output.split('\\'))
        plot = args.plot
        idscn.subtype(input_dir, outpath, plot)
    elif args.mode == 'dif':
        inpath = '/'.join(args.input.split('\\'))
        outpath = '/'.join(args.output.split('\\'))
        idscn.difference(inpath, outpath)
    else:
        parser.print_help()
