import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units
import NuRadioReco.utilities.io_utilities
import NuRadioReco.framework.electric_field
import NuRadioReco.detector.antennapattern
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import electricFieldParameters as efp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.utilities import trace_utilities, fft, bandpass_filter
import radiotools.helper as hp


class neutrino3DVertexReconstructor:

    def __init__(self, lookup_table_location):
        """
        Constructor for the vertex reconstructor

        Parameters
        --------------
        lookup_table_location: string
            path to the folder in which the lookup tables for the signal travel
            times are stored
        """
        self.__lookup_table_location = lookup_table_location
        self.__detector = None
        self.__lookup_table = {}
        self.__header = {}
        self.__channel_ids = None
        self.__station_id = None
        self.__channel_pairs = None
        self.__rec_x = None
        self.__rec_z = None
        self.__sampling_rate = None
        self.__passband = None
        self.__channel_pair = None
        self.__channel_positions = None
        self.__correlation = None
        self.__max_corr_index = None
        self.__current_ray_types = None
        self.__electric_field_template = None
        self.__distances = None
        self.__current_distance = None
        self.__pair_correlations = None
        self.__antenna_pattern_provider = NuRadioReco.detector.antennapattern.AntennaPatternProvider()
        self.__ray_types = [
            ['direct', 'direct'],
            ['reflected', 'reflected'],
            ['refracted', 'refracted'],
            ['direct', 'reflected'],
            ['reflected', 'direct'],
            ['direct', 'refracted'],
            ['refracted', 'direct'],
            ['reflected', 'refracted'],
            ['refracted', 'reflected']
        ]

    def begin(
            self,
            station_id,
            channel_ids,
            detector,
            template,
            distances=None,
            passband=None
    ):
        """
        General settings for vertex reconstruction

        Parameters
        -------------
        station_id: integer
            ID of the station to be used for the reconstruction
        channel_ids: array of integers
            IDs of the channels to be used for the reconstruction
        detector: Detector or GenericDetector
            Detector description for the detector used in the reconstruction
        """

        self.__detector = detector
        self.__channel_ids = channel_ids
        self.__station_id = station_id
        self.__channel_pairs = []
        for i in range(len(channel_ids) - 1):
            for j in range(i + 1, len(channel_ids)):
                if detector.get_antenna_type(station_id, channel_ids[i]) == detector.get_antenna_type(station_id, channel_ids[j]):
                    self.__channel_pairs.append([channel_ids[i], channel_ids[j]])
        self.__lookup_table = {}
        self.__header = {}
        self.__electric_field_template = template
        self.__sampling_rate = template.get_sampling_rate()
        self.__passband = passband
        if distances is None:
            self.__distances = np.arange(100, 3000, 200)
        else:
            self.__distances = distances
        for channel_id in channel_ids:
            channel_z = abs(detector.get_relative_position(station_id, channel_id)[2])
            if channel_z not in self.__lookup_table.keys():
                f = NuRadioReco.utilities.io_utilities.read_pickle('{}/lookup_table_{}.p'.format(self.__lookup_table_location, int(abs(channel_z))))
                self.__header[int(channel_z)] = f['header']
                self.__lookup_table[int(abs(channel_z))] = f['antenna_{}'.format(channel_z)]

    def run(
            self,
            event,
            station,
            det,
            debug=False
    ):
        theta_range = np.arange(0, 360.1, 2.5) * units.deg
        z_range = np.arange(-2700, -100, 20)

        theta_coords, z_coords = np.meshgrid(theta_range, z_range)
        distance_correlations = np.zeros(self.__distances.shape)
        full_correlations = np.zeros((len(self.__distances), len(z_range), len(theta_range)))

        if debug:
            plt.close('all')
            fig1 = plt.figure(figsize=(12, (len(self.__channel_pairs) // 2 + len(self.__channel_pairs) % 2)))
            fig2 = plt.figure(figsize=(12, 4 * (len(self.__distances) // 3 + len(self.__distances) % 2)))
        self.__pair_correlations = np.zeros((len(self.__channel_pairs), station.get_channel(self.__channel_ids[0]).get_number_of_samples() + self.__electric_field_template.get_number_of_samples() - 1))
        for i_pair, channel_pair in enumerate(self.__channel_pairs):
            channel_1 = station.get_channel(channel_pair[0])
            channel_2 = station.get_channel(channel_pair[1])
            antenna_response = trace_utilities.get_efield_antenna_factor(
                station=station,
                frequencies=self.__electric_field_template.get_frequencies(),
                channels=[channel_pair[0]],
                detector=det,
                zenith=90. * units.deg,
                azimuth=0,
                antenna_pattern_provider=self.__antenna_pattern_provider
            )[0]
            voltage_spec = (
                antenna_response[0] * self.__electric_field_template.get_frequency_spectrum() +
                antenna_response[1] * self.__electric_field_template.get_frequency_spectrum()
                        ) * det.get_amplifier_response(station.get_id(), channel_pair[0], self.__electric_field_template.get_frequencies())
            if self.__passband is not None:
                voltage_spec *= bandpass_filter.get_filter_response(self.__electric_field_template.get_frequencies(), self.__passband, 'butter', 10)
            voltage_template = fft.freq2time(voltage_spec, self.__sampling_rate)
            voltage_template /= np.max(np.abs(voltage_template))
            if self.__passband is None:
                corr_1 = hp.get_normalized_xcorr(channel_1.get_trace(), voltage_template)
                corr_2 = hp.get_normalized_xcorr(channel_2.get_trace(), voltage_template)
            else:
                corr_1 = hp.get_normalized_xcorr(channel_1.get_filtered_trace(self.__passband, 'butter', 10), voltage_template)
                corr_2 = hp.get_normalized_xcorr(channel_2.get_filtered_trace(self.__passband, 'butter', 10), voltage_template)
            correlation_product = np.zeros_like(corr_1)
            sample_shifts = np.arange(-len(corr_1) // 2, len(corr_1) // 2, dtype=int)
            toffset = sample_shifts / channel_1.get_sampling_rate()
            for i_shift, shift_sample in enumerate(sample_shifts):
                correlation_product[i_shift] = np.max(corr_1 * np.roll(corr_2, shift_sample))
            self.__pair_correlations[i_pair] = correlation_product
            if debug:
                ax1_1 = fig1.add_subplot(len(self.__channel_pairs) // 2 + len(self.__channel_pairs) % 2, 2,
                                         i_pair + 1)
                ax1_1.grid()
                ax1_1.plot(toffset, correlation_product)
        if debug:
            fig1.tight_layout()
            fig1.savefig('plots/correlations/correlation_{}_{}.png'.format(event.get_run_number(), event.get_id()))
        for i_dist, distance in enumerate(self.__distances):
            self.__current_distance = distance
            correlation_sum = np.zeros_like(theta_coords)

            for i_pair, channel_pair in enumerate(self.__channel_pairs):
                self.__correlation = self.__pair_correlations[i_pair]
                self.__channel_pair = channel_pair
                self.__channel_positions = [self.__detector.get_relative_position(self.__station_id, channel_pair[0]),
                                            self.__detector.get_relative_position(self.__station_id, channel_pair[1])]
                correlation_map = np.zeros_like(correlation_sum)
                for i_ray in range(len(self.__ray_types)):
                    self.__current_ray_types = self.__ray_types[i_ray]
                    correlation_map = np.maximum(self.get_correlation_array_2d(theta_coords, z_coords), correlation_map)
                correlation_sum += correlation_map
            distance_correlations[i_dist] = np.max(correlation_sum)
            full_correlations[i_dist] = correlation_sum

            if debug:
                ax2_1 = fig2.add_subplot(len(self.__distances) // 3 + len(self.__distances) % 3, 3, i_dist + 1)
                # cplot = ax2_1.pcolor(
                #     theta_coords / units.deg,
                #     z_coords,
                #     (correlation_sum)
                # )
                # plt.colorbar(cplot, ax=ax2_1)
                sim_vertex = None
                for sim_shower in event.get_sim_showers():
                    sim_vertex = sim_shower.get_parameter(shp.vertex)
                    break
                if sim_vertex is not None:
                    ax2_1.axvline(hp.get_normalized_angle(hp.cartesian_to_spherical(sim_vertex[0], sim_vertex[1], sim_vertex[2])[1]) / units.deg, color='r', linestyle='--', alpha=.5)
                    ax2_1.axhline(sim_vertex[2], color='r', linestyle='--', alpha=.5)
                ax2_1.set_title('d={:.0f}m'.format(distance))
        if debug:
            fig2.tight_layout()
            fig2.savefig('plots/direction_recos/direction_reco_{}_{}.png'.format(event.get_run_number(), event.get_id()))
            fig3 = plt.figure(figsize=(8, 16))
            ax3_1 = fig3.add_subplot(311)
            ax3_1.grid()
            sim_vertex = None
            for sim_shower in event.get_sim_showers():
                sim_vertex = sim_shower.get_parameter(shp.vertex)

                break
            corr_max = np.max(np.max(full_correlations, axis=1), axis=1)
            ax3_1.scatter(
                self.__distances,
                corr_max / np.max(corr_max)
            )
            corr_mean = np.mean(np.mean(full_correlations, axis=1), axis=1)
            ax3_1.scatter(
                self.__distances,
                corr_mean / np.max(corr_mean)
            )
            ax3_2 = fig3.add_subplot(312)
            ax3_2.grid()
            corr_max = np.max(np.max(full_correlations, axis=0), axis=0)
            ax3_2.scatter(
                theta_range / units.deg,
                corr_max / np.max(corr_max)
            )
            corr_mean = np.mean(np.mean(full_correlations, axis=0), axis=0)
            ax3_2.scatter(
                theta_range / units.deg,
                corr_mean / np.max(corr_mean)
            )
            ax3_3 = fig3.add_subplot(313)
            ax3_3.grid()
            corr_max = np.max(np.max(full_correlations, axis=0), axis=1)
            ax3_3.scatter(
                z_range,
                corr_max / np.max(corr_max)
            )
            corr_mean = np.mean(np.mean(full_correlations, axis=0), axis=1)
            ax3_3.scatter(
                z_range,
                corr_mean / np.max(corr_mean)
            )
            if sim_vertex is not None:
                ax3_1.axvline(np.sqrt(sim_vertex[0]**2 + sim_vertex[1]**2), color='k', linestyle=':')
                ax3_2.axvline(hp.get_normalized_angle(hp.cartesian_to_spherical(sim_vertex[0], sim_vertex[1], sim_vertex[2])[1]) / units.deg, color='k', linestyle=':')
                ax3_3.axvline(sim_vertex[2], color='k', linestyle=':')
            fig3.tight_layout()
            fig3.savefig('plots/distance_correlations/distance_correlation_{}_{}.png'.format(event.get_run_number(), event.get_id()))

            fig4 = plt.figure(figsize=(4, 12))
            fig5 = plt.figure(figsize=(4, 8))
            ax4_1 = fig4.add_subplot(311)
            ax5_1 = fig5.add_subplot(211)
            ax5_1.grid()
            corr_fit_threshold = .8 * np.max(full_correlations)
            flattened_corr = np.max(full_correlations, axis=2).T
            i_max = np.argmax(flattened_corr, axis=0)
            z_corr_mask = np.max(flattened_corr, axis=0) > corr_fit_threshold
            d_0, z_0 = np.meshgrid(self.__distances, z_range)
            ax4_1.pcolor(
                d_0,
                z_0,
                flattened_corr
            )
            ax4_1.grid()
            line_fit = np.polyfit(
                self.__distances[z_corr_mask],
                z_range[i_max][z_corr_mask],
                1
            )
            ax5_1.fill_between(
                self.__distances,
                self.__distances * line_fit[0] + line_fit[1] + 100,
                self.__distances * line_fit[0] + line_fit[1] - 100,
                color='k',
                alpha=.2
            )
            ax5_1.scatter(
                self.__distances[z_corr_mask],
                z_range[i_max][z_corr_mask]
            )
            ax5_1.scatter(
                self.__distances[~z_corr_mask],
                z_range[i_max][~z_corr_mask],
                c='k',
                alpha=.5
            )
            ax5_1.plot(
                self.__distances,
                self.__distances * line_fit[0] + line_fit[1],
                color='k',
                linestyle=':'
            )

            ax4_2 = fig4.add_subplot(312, projection='polar')
            ax5_2 = fig5.add_subplot(212, projection='polar')
            theta_0, d_0 = np.meshgrid(theta_range, self.__distances)
            flattened_corr = np.max(full_correlations, axis=1)
            i_max = np.argmax(flattened_corr, axis=1)
            ax4_2.pcolor(
                theta_0,
                d_0,
                flattened_corr
            )
            theta_corr_mask = np.max(flattened_corr, axis=1) >= corr_fit_threshold
            median_theta = np.median(theta_range[i_max][theta_corr_mask])
            ax5_2.scatter(
                theta_range[i_max][theta_corr_mask],
                self.__distances[theta_corr_mask]
            )
            ax5_2.scatter(
                theta_range[i_max][~theta_corr_mask],
                self.__distances[~theta_corr_mask],
                c='k',
                alpha=.5
            )
            ax5_2.plot(
                [median_theta, median_theta],
                [self.__distances[0], self.__distances[-1]],
                color='k',
                linestyle=':'
            )
            ax5_2.grid()
            ax4_2.grid()
            theta_0, z_0 = np.meshgrid(theta_range, z_range)
            ax4_3 = fig4.add_subplot(313)
            ax4_3.pcolor(
                theta_0 / units.deg,
                z_0,
                np.max(full_correlations, axis=0)
            )
            ax4_3.grid()
            if sim_vertex is not None:
                ax4_1.axhline(sim_vertex[2], color='r', linestyle='--', alpha=.5)
                ax4_1.axvline(np.sqrt(sim_vertex[0]**2 + sim_vertex[1]**2), color='r', linestyle='--', alpha=.5)
                ax5_1.axhline(sim_vertex[2], color='r', linestyle='--', alpha=.5)
                ax5_1.axvline(np.sqrt(sim_vertex[0]**2 + sim_vertex[1]**2), color='r', linestyle='--', alpha=.5)
                ax4_2.scatter(
                    [hp.cartesian_to_spherical(sim_vertex[0], sim_vertex[1], sim_vertex[2])[1]],
                    [np.sqrt(sim_vertex[0]**2 + sim_vertex[1]**2)],
                    c='r',
                    alpha=.5,
                    marker='+'
                )
                ax5_2.scatter(
                    [hp.cartesian_to_spherical(sim_vertex[0], sim_vertex[1], sim_vertex[2])[1]],
                    [np.sqrt(sim_vertex[0]**2 + sim_vertex[1]**2)],
                    c='r',
                    alpha=.5,
                    marker='+'
                )
                ax4_3.scatter(
                    [hp.get_normalized_angle(hp.cartesian_to_spherical(sim_vertex[0], sim_vertex[1], sim_vertex[2])[1]) / units.deg],
                    [(sim_vertex[2])],
                    c='r',
                    alpha=.5,
                    marker='+'
                )
            fig4.tight_layout()
            fig4.savefig('plots/correlation_slices/correlation_slice_{}_{}.png'.format(event.get_run_number(), event.get_id()))
            fig5.tight_layout()
            fig5.savefig('plots/maxima_paths/maxima_paths_{}_{}.png'.format(event.get_run_number(), event.get_id()))


    def get_correlation_array_2d(self, phi, z):
        """
        Returns the correlations corresponding to the different
        signal travel times between channels for the given positions.
        This is done by correcting for the distance of the channels
        from the station center and then calling
        self.get_correlation_for_pos, which does the actual work.

        Parameters:
        --------------
        x, z: array
            Coordinates of the points for which calculations are
            to be calculated. Correspond to the (r, z) pair
            of cylindrical coordinates.
        """
        channel_pos1 = self.__channel_positions[0]
        channel_pos2 = self.__channel_positions[1]
        x = self.__current_distance * np.cos(phi)
        y = self.__current_distance * np.sin(phi)
        d_hor1 = np.sqrt((x - channel_pos1[0])**2 + (y - channel_pos1[1])**2)
        d_hor2 = np.sqrt((x - channel_pos2[0])**2 + (y - channel_pos2[1])**2)
        res = self.get_correlation_for_pos(np.array([d_hor1, d_hor2]), z)
        return res

    def get_correlation_for_pos(self, d_hor, z):
        """
        Returns the correlations corresponding to the different
        signal travel times between channels for the given positions.

        Parameters:
        --------------
        d_hor, z: array
            Coordinates of the points for which calculations are
            to be calculated. Correspond to the (r, z) pair
            of cylindrical coordinates.
        """
        t1 = self.get_signal_travel_time(d_hor[0], z, self.__current_ray_types[0], self.__channel_pair[0])
        t2 = self.get_signal_travel_time(d_hor[1], z, self.__current_ray_types[1], self.__channel_pair[1])
        delta_t = t1 - t2
        delta_t = delta_t.astype(float)
        corr_index = self.__correlation.shape[0] / 2 + np.round(delta_t * self.__sampling_rate)
        corr_index[np.isnan(delta_t)] = 0
        mask = (corr_index > 0) & (corr_index < self.__correlation.shape[0]) & (~np.isinf(delta_t))
        corr_index[~mask] = 0
        res = np.take(self.__correlation, corr_index.astype(int))
        res[~mask] = 0
        return res

    def get_signal_travel_time(self, d_hor, z, ray_type, channel_id):
        """
        Calculate the signal travel time between a position and the
        channel

        Parameters:
        ------------
        d_hor, z: numbers or arrays of numbers
            Coordinates of the point from which to calculate the
            signal travel times. Correspond to (r, z) coordinates
            in cylindrical coordinates.
        ray_type: string
            Ray type for which to calculate the travel times. Options
            are direct, reflected and refracted
        channel_id: int
            ID of the channel to which the travel time shall be calculated
        """
        channel_pos = self.__detector.get_relative_position(self.__station_id, channel_id)
        channel_type = int(abs(channel_pos[2]))
        travel_times = np.zeros_like(d_hor)
        mask = np.ones_like(travel_times).astype(bool)
        i_x = np.array(np.round((-d_hor - self.__header[channel_type]['x_min']) / self.__header[channel_type]['d_x'])).astype(int)
        mask[i_x > self.__lookup_table[channel_type][ray_type].shape[0] - 1] = False
        i_z = np.array(np.round((z - self.__header[channel_type]['z_min']) / self.__header[channel_type]['d_z'])).astype(int)
        mask[i_z > self.__lookup_table[channel_type][ray_type].shape[1] - 1] = False
        i_x[~mask] = 0
        i_z[~mask] = 0
        travel_times = self.__lookup_table[channel_type][ray_type][(i_x, i_z)]
        travel_times[~mask] = np.nan
        return travel_times
