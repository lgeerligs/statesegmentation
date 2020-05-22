from numpy import concatenate, cov, ndarray, nonzero, ones, triu, unique, zeros
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, ttest_ind
from typing import Optional


class GSBS:
    def __init__(self, kmax: int, x: ndarray, dmin: int = 1, y: Optional[ndarray] = None) -> None:
        """Given an ROI timeseries, this class uses a greedy search algorithm (GSBS) to  
        segment the timeseries into neural states with stable activity patterns.
        GSBS identifies the timepoints of neural state transitions, while t-distance is used
        to determine the optimal number of neural states.
        
        You can find more information about the method here:
        Geerligs L., van Gerven M., Güçlü U (2020) Detecting neural state transitions underlying event segmentation
        biorXiv. https://doi.org/10.1101/2020.04.30.069989 
        
        Arguments:
            kmax {int} -- the maximal number of neural states that should be estimated in the greedy search
            x {ndarray} -- a multivoxel ROI timeseries - timepoint by voxels array
                           

        Keyword Arguments:
            dmin {int} -- the number of TRs around the diagonal of the time by time correlation matrix that are not taken 
                            into account in the computation of the t-distance metric (default: {1})
            y {Optional[ndarray]} -- a multivoxel ROI timeseries - timepoint by voxels array
                                      if y is given, the t-distance will be based on cross-validation, 
                                      such that the state boundaries are identified using the data in x and the 
                                      optimal number of states is identified using the data in y. If y is not given
                                      the state boundaries and optimal number of states are both based on x. (default: {None})
        """
        self.x = x
        self.y = y
        self.kmax = kmax
        self.dmin = dmin
        self._argmax = None
        self._bounds = zeros(self.x.shape[0], int)
        self._deltas = zeros(self.x.shape[0], bool)
        self._tdists = zeros(self.kmax + 1, float)

    @property
    def bounds(self) -> ndarray:
        """

        Returns:
            ndarray -- array with length == number of timepoints, where a zero indicates no state transition
            at a particular timepoint and a higher number indicates a state transition. State transitions
            are numbered in the order in which they are detected in GSBS (stronger boundaries tend
            to be detected first). 
        """
        assert self._argmax is not None
        return self._bounds * self.deltas

    @property
    def deltas(self) -> ndarray:
        """

        Returns:
            ndarray -- array with length == number of timepoints, where a zero indicates no state transition
            at a particular timepoint and a one indicates a state transition.
        """
        assert self._argmax is not None
        return self._bounds <= self._argmax

    @property
    def strengths(self) -> ndarray:
        """

        Returns:
            ndarray -- array with length == number of timepoints, where a zero indicates no state transition
            at a particular timepoint and another value indicates a state transition. The numbers indicate
            the strength of a state transition, as indicated by the correlation-distance between neural
            activity patterns in consecutive states. 
        """
        assert self._argmax is not None
        deltas = self.deltas
        states = self._states(deltas, self.x)
        states_unique = unique(states)
        pcorrs = zeros(len(states_unique) - 1, float)
        xmeans = zeros((len(states_unique), self.x.shape[1]), float)

        for state in states_unique:
            xmeans[state] = self.x[state == states].mean(0)
            if state > 0:
                pcorrs[state - 1] = pearsonr(xmeans[state], xmeans[state - 1])[0]

        strengths = zeros(deltas.shape, float)
        strengths[deltas] = 1 - pcorrs

        return strengths

    def fit(self) -> None:
        """This function performs the GSBS and t-distance computations to determine
        the location of state boundaries and the optimal number of states. 
        """
        i = triu(ones(self.x.shape[0], bool), self.dmin)
        z = GSBS._zscore(self.x)
        r = cov(z)[i]
        t = GSBS._zscore((r if self.y is None else cov(GSBS._zscore(self.y))[i])[None])[0]

        for k in range(2, self.kmax + 1):
            states = self._states(self._deltas, self.x)
            wdists = self._wdists(self._deltas, states, self.x, z)
            argmax = wdists.argmax()
            states = concatenate(states[:argmax], states[argmax:] + 1)[:, None]
            diff, same = (lambda c: (c == 1, c == 0))(cdist(states, states, "cityblock")[i])
            self._bounds[argmax] = k
            self._deltas[argmax] = True
            self._tdists[k] = 0 if sum(same) < 2 else ttest_ind(t[diff], t[same], equal_var=False)[0]

        self._argmax = self._tdists.argmax()

    @staticmethod
    def _states(deltas: ndarray, x: ndarray) -> ndarray:
        states = zeros(x.shape[0], int)
        for i, delta in enumerate(deltas[1:]):
            states[i + 1] = states[i] + 1 if delta else states[i]

        return states

    @staticmethod
    def _wdists(deltas: ndarray, states: ndarray, x: ndarray, z: ndarray) -> ndarray:
        xmeans = zeros(x.shape, float)
        wdists = zeros(x.shape[0], float)

        for state in map(lambda s: s == states, unique(states)):
            xmeans[state] = x[state].mean(0)

        for i in range(1, x.shape[0]):
            if deltas[i] == 0:
                state = nonzero(states[i] == states)[0]
                xmean = xmeans[state]  # ಠ_ಠ
                xmeans[state[0]: i] = x[state[0]: i].mean(0)
                xmeans[i: state[-1] + 1] = x[i: state[-1] + 1].mean(0)
                wdists[i] = xmeans.shape[1] * (GSBS._zscore(xmeans) * z).mean() / (xmeans.shape[1] - 1)
                xmeans[state] = xmean

        return wdists

    @staticmethod
    def _zscore(x: ndarray) -> ndarray:
        return (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
