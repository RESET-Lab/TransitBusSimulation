import numpy as np
import pandas as pd
import time
import geopandas as gpd
from matplotlib import colors
import matplotlib.pyplot as plt
from pathlib import Path
from collections import namedtuple

from fastsim import simdrive, vehicle, cycle

# Mexicali Shapefiles. Source: https://www.mexicali.gob.mx/sitioimip/geovisor/layers/geonode:distrito_pducp25/metadata_detail
SHAPEFILE_PATH = '../Data/LOGIOS Data/Mexicali Shapefiles/distrito_pducp25.shp'

# Unit conversion constants
m_per_km = 1000
km_per_mile = 1.609344
s_per_hour = 3600

def run_fastsim(drive_cycle_path, veh_jit, segment_averaged=False):
    '''
    Runs FASTSim on the drive cycle (specified by csv files at the given path) using the specified
    vehicle.

    :param path: Path object specifying the location of the directory containting the drive cycle
        csv files (alphabetically ordered).
    :param veh_jit: FASTSim numba Vehicle instance.
    :param segment_averaged: If True, modifies the raw drive cycle to have segment-averaged value for
        speed and slope.
    :returns: FASTSim dict of summary output variables.
    '''
    # Load and instantiate Cycle
    cyc_dict = load_drive_cycle(drive_cycle_path, num_files=None, visualize=False, segment_averaged=segment_averaged)
    cyc = cycle.Cycle(cyc_dict=cyc_dict)
    cyc_jit = cyc.get_numba_cyc()

    # Run FASTSim
    # Note: when the simulation keeps running beyond when SOC hits 0% we get the
    # 'Warning: There is a problem with conservation of energy' message.
    t0 = time.time()
    sim_drive = simdrive.SimDriveJit(cyc_jit, veh_jit)
    sim_drive.sim_drive()
    sim_drive_post = simdrive.SimDrivePost(sim_drive)
    output = sim_drive_post.get_output()
    print(f'Time to run simulation: {time.time() - t0:.2e} s')

    # Compute and attach a few more values on the output
    output['speed'] = cyc.cycMps
    output['essKwOutAch'] = sim_drive.essKwOutAch

    # Cumulative distance series
    dist_per_step = sim_drive.distMeters
    output['cumDistKm'] = cum_series(dist_per_step) / m_per_km

    # Cumulative kWh series: i.e. cumulative energy required to get to a given timestep.
    # essKwOutAch is a series of kW output per second
    sign_toggle = (sim_drive.essKwOutAch > 0).astype(int)
    # Strip out negative power output measurements (from regenerative breaking)
    pos_essKwOutAch = sim_drive.essKwOutAch * sign_toggle
    output['cumKwh'] = cum_series(pos_essKwOutAch) / s_per_hour

    # Distance when we hit 0% S.0.C.
    num_steps = len(output['soc'])
    minSoc_time = num_steps - 1
    for i in range(num_steps):
        if abs(output['soc'][i]) <= 10**(-5): # TODO smarter search if too slow
            minSoc_time = i
            break
    if minSoc_time == num_steps-1:
        print('S.O.C. 0% never reached!') # TODO do something smarter, could often not reach 0 in general
    output['rangeKm'] = output['cumDistKm'][minSoc_time]
    # print(f'Bus range before 0% S0C: {output['rangeKm']:.02f} km')
    # print(f'Total energy capacity required: {output["cumKwh"][minSoc_time]:.02f} kWh')
    return output

def load_drive_cycle(path, num_files=None, visualize=False, segment_averaged=False):
    '''
    Loads a drive cycle, segmented into multiple csv files, into memory.

    :param path: Path object specifying the location of the directory containting the drive cycle
        csv files (alphabetically ordered).
    :param num_files: The max number of csv files (each representing a portion of the full drive cycle)
        to load and concatenate. If None, all files will be loaded.
    :param vizualize: Plot map visualizations of the drive cycle route (segment by segment).
    :param segment_averaged: If True, modifies the raw drive cycle to have segment-averaged value for
        speed and slope.
    :returns: A drive cycle dict structure that can be passed to fastsim's Cycle constructor.
    '''
    # Load LOGIOS drive cycles
    sorted_dc_files = sorted(path.iterdir(), key=lambda f: str(f))

    # Load Mexicali map shapefile
    shapefile_path = Path(SHAPEFILE_PATH)
    mexicali_map = gpd.read_file(shapefile_path) # At this point GeoDataFrame's CRS is epsg:32611\
    mexicali_4326 = mexicali_map.to_crs("EPSG:4326") # Change CRS

    # Color the bus route's directionality: Green at start, red at end.
    cm = colors.LinearSegmentedColormap.from_list('namewhy', colors=['green', 'red'], N=10000)

    # Visualize the route data in each CSV file
    if visualize:
        for file in sorted_dc_files[0:num_files]:
            ax = mexicali_4326.plot(color='white', edgecolor='black')

            df = pd.read_csv(file)
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
            gdf.plot(ax=ax, cmap=cm)

            plt.title(f'{file.name}')
            plt.show()

    # Concatenate the drive cycle dataframes into the full day's drive cycle
    cyc_dfs = [pd.read_csv(file) for file in sorted_dc_files[0:num_files]]
    cyc_df = pd.concat(cyc_dfs, axis=0)

    # Overwrite index and time to be monotonically increasing. The individual cycle files were each
    # re-indexed to 0 by LOGIOS.
    cyc_df.Time[:] = range(0,len(cyc_df.Time))
    cyc_df.set_index('Time', inplace=True)
    
    if segment_averaged:
        cyc_df = segment_average(cyc_df)
    
    # LOGIOS speed measurements are km/h. Convert to m/s for fastsim
    cyc_df.Speed *= 1000/3600

    # Create a drive cycle dict in a format that FASTSim understands
    cyc_dict = {
        'cycSecs': cyc_df.index.to_numpy(),
        'cycMps': cyc_df.Speed.to_numpy(),
        'cycGrade': cyc_df.Slope.to_numpy(),
    }
    return cyc_dict

# Given a timeseries of measurements (e.g. timeseries of 1-second distance travelled)
# Return a new timseries of cumulative measurements (e.g. timeseries of total distance)
# TODO depricate this in favor of pandas series cumsum()
def cum_series(x):
    n = len(x)
    y = np.zeros(n)
    y[0] = x[0]
    for i in range(1, n):
        y[i] += y[i-1] + x[i]
    return y

SEGMENT_LEN_M = 50 # meters
def segment_average(raw_cyc_df):
    '''
    Given a raw drive cycle, return a segment-averaged drive cycle.
    
    Assumptions about raw drive cycle data from LOGIOS
    - Measurements occur at regular 1-second intervals with no gaps
    - Speed column is in units of km/hour
    '''
    # Convert Speed from km/h to m/s
    logios_speed_mps = raw_cyc_df.Speed * 1000/3600
    
    # Each speed measurement is actually a distance measurement bc mps * 1s = mps
    dx = logios_speed_mps
    
    # Calculate the cumulative distance traveled
    cumDistKm = dx.cumsum() / 1000
    
    # Break route into SEGMENT_LEN_M-meter segments
    
    # Detect segment breakpoints
    # A series of (time, cumDist) tuples marking the segment starts
    Segment = namedtuple('Segment', ['time', 'cumDist'])
    dx_breakpoints = [Segment(0,0)]
    for t in range(1, len(raw_cyc_df)):
        if cumDistKm.iloc[t] >= len(dx_breakpoints) * SEGMENT_LEN_M / m_per_km:
            dx_breakpoints.append(Segment(t, cumDistKm.iloc[t]))
    print(f'There are {len(dx_breakpoints)} {SEGMENT_LEN_M}-meter segments in this {cumDistKm.iloc[-1]:.01f} km route')

    # Create segment-averaged drive cycle
    avg_df = raw_cyc_df.copy()
    for i in range(len(dx_breakpoints)-1):
        start, end = dx_breakpoints[i].time, dx_breakpoints[i+1].time
        r = range(start, end)
        # Calculate average speed and slope for this segment
        avg_speed = raw_cyc_df.Speed[r.start:r.stop].mean()
        avg_slope = raw_cyc_df.Slope[r.start:r.stop].mean()
        # Write the average value for every measurement in this segment
        avg_df.Speed[r.start:r.stop] = np.array([avg_speed for i in r])
        avg_df.Slope[r.start:r.stop] = np.array([avg_slope for i in r])
    return avg_df

def run_tests():
    '''
    Basic tests.
    '''
    assert all(cum_series(np.array([1,1,1])) == np.array([1,2,3]))
