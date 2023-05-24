import numpy as np
from collections import Counter
from sklearn import metrics
from typing import Mapping, Sequence, Dict


class Evaluator():
    """ Evaluate the accuracy and fairness on demographic parity and equal opportunity """

    def __init__(self, s: np.ndarray, name: str):
        """
        :param s: numpy array containing all the sensitive label (categorical or binary)
        """
        self.s = s
        self.name = name

        self.all_grp = sorted(set(self.s))
        self.grp_num = Counter(self.s)
        self.normalized_factor = len(self.all_grp) * (len(self.all_grp) - 1) / 2.

    @staticmethod
    def acc(y, pred) -> float:
        return metrics.accuracy_score(y, pred)

    def dp(self, grp_dict: Mapping) -> float:
        """ Demographic parity """

        if len(self.all_grp) == 2:
            dp = sum(grp_dict[1.]["pred"]) / self.grp_num[1.] - sum(grp_dict[0.]["pred"]) / self.grp_num[0.]
        else:
            dp = 0.
            all_grp = list(self.all_grp)
            for i, g_1 in enumerate(self.all_grp):
                for g_2 in all_grp[i + 1:]:
                    gap = sum(grp_dict[g_1]["pred"]) / self.grp_num[g_1] \
                          - sum(grp_dict[g_2]["pred"]) / self.grp_num[g_2]
                    dp += abs(gap)

            dp /= self.normalized_factor

        return dp

    def eop(self, grp_dict: Mapping) -> float:
        """ Equal opportunity """

        for g in self.all_grp:
            grp_dict[g]["pred_cond_pos"] = [e for i, e in enumerate(grp_dict[g]["pred"]) if grp_dict[g]["y"][i] == 1]

        if len(self.all_grp) == 2:
            eop = sum(grp_dict[1.]["pred_cond_pos"]) / len(grp_dict[1.]["pred_cond_pos"]) \
                  - sum(grp_dict[0.]["pred_cond_pos"]) / len(grp_dict[0.]["pred_cond_pos"])
        else:
            eop = 0.
            all_grp = list(self.all_grp)
            for i, g_1 in enumerate(self.all_grp):
                for g_2 in all_grp[i + 1:]:
                    gap = sum(grp_dict[g_1]["pred_cond_pos"]) / len(grp_dict[g_1]["pred_cond_pos"]) \
                          - sum(grp_dict[g_2]["pred_cond_pos"]) / len(grp_dict[g_2]["pred_cond_pos"])
                    eop += abs(gap)

            eop /= self.normalized_factor

        return eop

    def gen_grp_dict(self, y, pred) -> Mapping:
        grp_dict = {g: {} for g in self.all_grp}
        for g in self.all_grp:
            grp_dict[g]["y"] = [e for i, e in enumerate(y) if self.s[i] == g]
            grp_dict[g]["pred"] = [e for i, e in enumerate(pred) if self.s[i] == g]

        return grp_dict

    def __call__(self, y: Sequence, pred: Sequence) -> Dict:
        assert len(y) == len(pred)

        grp_dict = self.gen_grp_dict(y, pred)

        dp = self.dp(grp_dict)
        eop = self.eop(grp_dict)
        overall_acc = self.acc(y, pred)

        print("-" * 30, "Results on %s" % self.name)
        group_acc = []
        for g in self.all_grp:
            g_acc = self.acc(grp_dict[g]["y"], grp_dict[g]["pred"])
            group_acc.append(g_acc)
            print("Grp. %d - #instance: %d; #pos. pred: %d; Acc.: %.6f" %
                  (g, self.grp_num[g], sum(grp_dict[g]["pred"]), g_acc))
        print("Overall acc.: %.6f; Demographic parity: %.6f; Equal opportunity: %.6f" % (overall_acc, dp, eop))

        res = {"overall_acc": overall_acc, "dp": dp, "eop": eop}
        res.update({"grp_%s_acc" % i: acc for i, acc in enumerate(group_acc)})

        return res
