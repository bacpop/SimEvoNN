## This script will calculate summary statistics of phylogenetic trees

### Based on from DOI:10.1371/journal.pcbi.1005416

"""
Notation Description
max_H Sum of the branch lengths between the root and its farthest leaf
min_H Sum of the branch lengths between the root and its closest leaf
a_BL_mean Mean length of all branches
a_BL_median Median length of all branches
a_BL_var Variance of the lengths of all branches
e_BL_mean Mean length of external branches
e_BL_median Median length of external branches
e_BL_var Variance of the lengths of external branches
i_BL_mean_[k]· Piecewise mean length of internal branches
i_BL_median_[k]· Piecewise median length of internal branches
i_BL_var_[k]· Piecewise variance of the lengths of internal branches
ie_BL_mean_[k]  Ratio of the piecewise mean length of internal branches over the mean length of
external branches
ie_BL_median_[k]  Ratio of the piecewise median length of internal branches over the median length of
external branches
ie_BL_var_[k]  Ratio of the piecewise variance of the lengths of internal branches over the variance
of the lengths of external branches
colless Sum for each internal node of the absolute difference between the number of leaves on
the left side and the number of leaves on the right side [47]
sackin Sum for each leaf of the number of internal nodes between the leaf and the root [48]
WD_ratio Ratio of the maximal width (W) over the maximal depth (D), where the depth of a node
characterizes the number of branches that lies between it and the root, and the width wd
of a tree at a depth level d is the number of nodes that have the same depth d [49]
Δw Maximal difference in width Dw  maxD􀀀 1
d0 jwd 􀀀 wd1j [49]
max_ladder Maximal number of internal nodes in a ladder which is a chain of connected internal
nodes each linked to a single leaf, divided by the number of leaves [49]
IL_nodes Proportion of internal nodes In Ladders [49]
staircaseness_1 Proportion of imbalanced internal nodes that have different numbers of leaves between
the left and the right side [49]
staircaseness_2 Mean ratio of the minimal number of leaves on a side over the maximal number of leaves
on a side, for each internal node [49, 50]
"""

from ete3 import Tree
from ete3.parser.newick import NewickError


class PhyloTree:
    def __init__(self, tree_path, tree_format=5, tree=None):
        """
        :param tree: tree file
        :param tree_format: values may be needed by ete3 package
        """
        self.tree_path = tree_path
        self.tree_format = tree_format
        self.tree = tree if tree is not None else self.read_tree(self.tree_path)


        self.max_H = None
        self.min_H = None
        self.a_BL_mean = None
        self.a_BL_median = None
        self.a_BL_var = None

        self.tree_stats= None

        ## Collect all distances from root to leaf
        self._leaf_distances = None


    def update_tree_stats(self):
        self.tree_stats = {
            "max_H": self.max_H,
            "min_H": self.min_H,
            "a_BL_mean": self.a_BL_mean,
            "a_BL_median": self.a_BL_median
        }
    def get_tree_stats(self):
        self._leaf_distances = [self.tree.get_distance(leaf) for leaf in self.tree.get_leaves()]
        self._get_min_distance_from_root()
        self._get_max_distance_from_root()
        self._get_bl_mean()
        self._get_bl_median()
        self.update_tree_stats()
        return self.tree_stats

    def _get_min_distance_from_root(self):
        self.min_H = self.tree.get_closest_leaf()[1]
        #self.min_H = min(self._leaf_distances)
    def _get_max_distance_from_root(self):
        self.max_H = self.tree.get_farthest_leaf()[1]
        #self.max_H = max(self._leaf_distances)

    def _get_bl_mean(self):
        self.a_BL_mean = sum(self._leaf_distances) / len(self._leaf_distances)

    def _get_bl_median(self):
        self.a_BL_median = sorted(self._leaf_distances)[len(self._leaf_distances) // 2]

    @staticmethod
    def read_tree(newick_tree): ###Taken from PhyloDeep model code
        """ Tries all nwk formats and returns an ete3 Tree

        :param newick_tree: str, a tree in newick format
        :return: ete3.Tree
        """
        tree = None
        for f in (3, 2, 5, 0, 1, 4, 6, 7, 8, 9):
            try:
                tree = Tree(newick_tree, format=f)
                break
            except NewickError as e:
                print(e)
                continue
        if not tree:
            raise ValueError('Could not read the tree {}. Is it a valid newick?'.format(newick_tree))
        return tree

    def read_tree_file(self): ### Taken from PhyloDeep model code
        with open(self.tree_path, 'r') as f:
            nwk = f.read().replace('\n', '').split(';')
            if nwk[-1] == '':
                nwk = nwk[:-1]
        if not nwk:
            raise ValueError('Could not find any trees (in newick format) in the file {}.'.format(self.tree_path))
        if len(nwk) > 1:
            raise ValueError('There are more than 1 tree in the file {}. Now, we accept only one tree per inference.'.format(self.tree_path))
        return self.read_tree(nwk[0] + ';')

    def save_stats(self, path):
        import json
        with open(path, "w") as f:
            json.dump(self.tree_stats, f)

    def save_tree(self, path):
        self.tree.render(path)



#instr = "((Sample1:0.0,Sample0:0.09333496093749999):0.1,(Sample2:0.1,Sample3:0.0):0.1):1.0;"
#phyltree = PhyloTree(
#    tree_path="MAPLE_FWsim_out_tree.tree"
#)
#phyltree.get_tree_stats()
#print(phyltree.tree_stats)
#phyltree.tree.render("tryandcry.png")
