

def main(args):
    from lib.phylogenetic_tree import PhyloTree

    pt_obj = PhyloTree(
        tree_path=args.infile_path
    )

    pt_obj.get_summary_statistics()
    pt_obj.save_tree(args.output_path)
    pt_obj.save_stats(args.output_path.replace(".png", "_stats.json"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile_path", type=str, required=True, help="Path to the tree file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save tree file (ending .png) and stats")
    args = parser.parse_args()
    main(args)
