import numpy as np
#import scipy.signal
#import matplotlib.pyplot as plt
#from NuRadioReco.utilities import units, fft
#import NuRadioReco.utilities.io_utilities
#import NuRadioReco.framework.electric_field
#from NuRadioReco.framework.parameters import stationParameters as stnp
#import radiotools.helper as hp
#import NuRadioMC.SignalProp.analyticraytracing
#import NuRadioMC.utilities.medium
#import NuRadioMC.SignalGen.askaryan
from NuRadioReco.modules.base import module#, dont forget to inherit from module Keith

class neutrinoBancroftVertexReconstructor():

    def __init__(self):
        self.__detector = None
        self.__channel_ids = None
        self.__station_id = None
        self.__output_path = None
        self.__channel_positions = []

    def begin(self, station_id, channel_ids, detector, output_path=None):
        self.__detector = detector
        self.__channel_ids = channel_ids
        self.__station_id = station_id
        self.__output_path = output_path

        for cid in channel_ids:
            channel_position = detector.get_relative_position(station_id, cid, mode='channel')
            self.__channel_positions.append(channel_position)

        self.__channel_positions = np.array(self.__channel_positions)

        print(self.__channel_positions)

    def run(self, event, station, debug=False):
        reference_channel = station.get_channel(self.__channel_ids[0])
        reference_trace = reference_channel.get_trace()
        sampling_rate = reference_channel.get_sampling_rate()

        tdoas = [(np.argmax(np.correlate(reference_trace, station.get_channel(cid).get_trace(), mode='full')) 
                   - len(reference_trace) + 1) / sampling_rate for cid in self.__channel_ids]
        tdoas -= min(tdoas)

        reconstructed_position = self.__bancroft(self.__channel_positions, np.array(tdoas).reshape(-1, 1))

        print("Position: ", reconstructed_position)
        print("TDoA's: ", tdoas) # In what units are the times?

    def __bancroft(self, channel_positions, tdoas):
        def objective_function(test_point, tdoas):
            pseudoranges = np.linalg.norm(self.__channel_positions - test_point, axis=1) / 3e+8
            return np.sum((pseudoranges - tdoas) ** 2)

        def lorentzian_inner_product(v, w):
            return np.sum(v * (w @ M), axis=-1)

        M = np.diag([1, 1, 1, -1]) # Minkowski matrix

        pseudorange_matrix = np.append(self.__channel_positions, tdoas * 3e+8, axis=1)
        b_matrix = 0.5 * lorentzian_inner_product(pseudorange_matrix, pseudorange_matrix)
        augmented_pseudorange_matrix = np.linalg.solve(np.dot(pseudorange_matrix.T, pseudorange_matrix),
                                                       np.dot(pseudorange_matrix.T, np.ones(len(self.__channel_positions))))
        inverse_pseudorange_matrix = np.linalg.solve(np.dot(pseudorange_matrix.T, pseudorange_matrix),
                                                   np.dot(pseudorange_matrix.T, b_matrix))

        solutions = []
        for root in np.roots([lorentzian_inner_product(augmented_pseudorange_matrix, augmented_pseudorange_matrix),
                             (lorentzian_inner_product(augmented_pseudorange_matrix, inverse_pseudorange_matrix) - 1) * 2,
                              lorentzian_inner_product(inverse_pseudorange_matrix, inverse_pseudorange_matrix),
                             ]):
            X, Y, Z, T = M @ np.linalg.solve((pseudorange_matrix.T @ pseudorange_matrix), (pseudorange_matrix.T @ (root * np.ones(len(self.__channel_positions)) + b_matrix)))
            solutions.append(np.array([X,Y,Z]))

        return min(solutions, key = lambda pt: objective_function(pt, tdoas)) 
