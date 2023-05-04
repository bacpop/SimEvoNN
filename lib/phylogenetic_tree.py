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
import numpy as np
from ete3 import Tree
from ete3.parser.newick import NewickError
from lib import sumstats

class PhyloTree:

    tree_stats_idx = {
        "max_H": 0,
        "min_H": 1,
        "a_BL_mean": 2,
        "a_BL_median": 3,
        "a_BL_var": 4,
        "e_BL_mean": 5,
        "e_BL_median": 6,
        "e_BL_var": 7,
        "i_BL_mean_1": 8,
        "i_BL_median_1": 9,
        "i_BL_var_1": 10,
        "ie_BL_mean_1": 11,
        "ie_BL_median_1": 12,
        "ie_BL_var_1": 13,
        "i_BL_mean_2": 14,
        "i_BL_median_2": 15,
        "i_BL_var_2": 16,
        "ie_BL_mean_2": 17,
        "ie_BL_median_2": 18,
        "ie_BL_var_2": 19,
        "i_BL_mean_3": 20,
        "i_BL_median_3": 21,
        "i_BL_var_3": 22,
        "ie_BL_mean_3": 23,
        "ie_BL_median_3": 24,
        "ie_BL_var_3": 25,
        "colless": 26,
        "sackin": 27,
        "WD_ratio": 28,
        "delta_w": 29,
        "max_ladder": 30,
        "IL_nodes": 31,
        "staircaseness_1": 32,
        "staircaseness_2": 33,
        "tree_size": 34,
    }
    def __init__(self, tree_path, tree_format=5, tree=None):
        """
        :param tree: tree file
        :param tree_format: values may be needed by ete3 package
        """
        self.tree_path = tree_path
        self.tree_format = tree_format
        self.tree = tree if tree is not None else self.read_tree(self.tree_path)
        self.tree_stats = np.zeros(len(self.tree_stats_idx))

        ### Tree statistics
        #self.max_H, self.min_H = None, None
        #self.all_bl_mean, self.all_bl_median, self.all_bl_var, self.ext_bl_mean, self.ext_bl_median, self.ext_bl_var = None, None, None, None, None, None

    def reset_tree(self):
        self.tree_stats = np.zeros(len(self.tree_stats_idx))
        self.tree = self.read_tree(self.tree_path)

    def get_tree_stats(self):
        return self.tree_stats

    def get_summary_statistics(self):
        """Rescales all trees from tree_file so that mean branch length is 1,
        then encodes them into summary statistics representation

        :param tree_input: ete3.Tree, on which the summary statistics will be computed
        :param sampling_proba: float, presumed sampling probability for all the trees
        :return: pd.DataFrame, encoded rescaled input trees in the form of summary statistics and float, a rescale factor
        """
        # local copy of input tree
        tree = self.tree.copy()

        # compute the rescale factor
        #rescale_factor = rescale_tree(tree, target_avg_length=TARGET_AVG_BL)

        # add accessory attributes
        #name_tree(tree)
        max_depth = sumstats.add_depth_and_get_max(tree)
        self._add_dist_to_root(tree)
        sumstats.add_ladder(tree)
        sumstats.add_height(tree)

        # compute summary statistics based on branch lengths
        summaries = []
        #self.max_H, self.min_H = sumstats.tree_height(tree)
        #self.all_bl_mean, self.all_bl_median, self.all_bl_var, self.ext_bl_mean, self.ext_bl_median, self.ext_bl_var = sumstats.branches(tree)
        summaries.extend(sumstats.tree_height(tree)) # max_H, min_H
        summaries.extend(sumstats.branches(tree)) # all_bl_mean, all_bl_median, all_bl_var, ext_bl_mean, ext_bl_median, ext_bl_var
        summaries.extend(sumstats.piecewise_branches(tree, summaries[0], summaries[5], summaries[6], summaries[7])) #i_bl_mean_1, i_bl_median_1, i_bl_var_1, ie_bl_mean_1, ie_bl_median_1, ie_bl_var_1, i_bl_mean_2, i_bl_median_2, i_bl_var_2, ie_bl_mean_2, ie_bl_median_2, ie_bl_var_2, i_bl_mean_3, i_bl_median_3, i_bl_var_3, ie_bl_mean_3, ie_bl_median_3, ie_bl_var_3
        summaries.append(sumstats.colless(tree))
        summaries.append(sumstats.sackin(tree))
        summaries.extend(sumstats.wd_ratio_delta_w(tree, max_dep=max_depth))
        summaries.extend(sumstats.max_ladder_il_nodes(tree))
        summaries.extend(sumstats.staircaseness(tree))

        # compute summary statistics based on LTT plot
        ##TODO: look-up what LTT plot is and if we need it
        #ltt_plot_matrix = sumstats.ltt_plot(tree)
        #summaries.extend(sumstats.ltt_plot_comput(ltt_plot_matrix))

        # compute LTT plot coordinates

        #summaries.extend(sumstats.coordinates_comp(ltt_plot_matrix))

        # compute summary statistics based on transmission chains (order 4):
        ##TODO: look-up what transmission chains are and if we need it
        summaries.append(len(tree))
        #summaries.extend(sumstats.compute_chain_stats(tree, order=4))
        #summaries.append(sampling_proba)

        self.tree_stats = np.array(summaries, dtype=np.float16)

        return self.tree_stats

    @staticmethod
    def _add_dist_to_root(tre):
        """
        Add distance to root (dist_to_root) attribute to each node
        :param tre: ete3.Tree, tree on which the dist_to_root should be added
        :return: void, modifies the original tree
        """

        for node in tre.traverse("preorder"):
            if node.is_root():
                node.add_feature("dist_to_root", 0)
            elif node.is_leaf():
                node.add_feature("dist_to_root", getattr(node.up, "dist_to_root") + node.dist)
                # tips_dist.append(getattr(node.up, "dist_to_root") + node.dist)
            else:
                node.add_feature("dist_to_root", getattr(node.up, "dist_to_root") + node.dist)
                # int_nodes_dist.append(getattr(node.up, "dist_to_root") + node.dist)
        return None
    def get_tree_stat(self, stat_name:str):
        return self.tree_stats[self.tree_stats_idx[stat_name]]

    @property
    def stat_names(self):
        return self.tree_stats_idx.keys()

    @property
    def stat_indices(self):
        return self.tree_stats_idx

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

    def _convert_data_to_dict(self):
        return {key_name: float(self.tree_stats[self.tree_stats_idx[key_name]]) for key_name in self.tree_stats_idx.keys()}

    @property
    def tree_stats_dict(self):
        return self._convert_data_to_dict()

    def save_stats(self, path):
        import json
        with open(path, "w") as f:
            json.dump(self.tree_stats_dict, f)

    def save_tree(self, path):
        self.tree.render(path)

