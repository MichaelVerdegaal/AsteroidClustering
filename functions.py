from math import sqrt, pow, cos, sin, tan, atan

import numpy as np
import pandas as pd
import ujson
import pendulum

START_ORBIT_TIMESTAMP = '2021-01-01T00:00:00+00:00'  # Orbit day zero
START_ARRIVAL_TIMESTAMP = '2021-04-17T14:00:00+00:00'  # Adalia day zero ("The Arrival")
# Astronomical Unit multiplier. 1 point = 1 million kilometer. Used to arrive at correct xyz pos.
AU_MULTIPLIER = 149.597871


def position_at_adalia_day(a: float, e: float, i: float, o: float, w: float, m: float, aday: int):
    """
    Calculate the xyz coordinates of an asteroid at a certain adalia day.

    This function is a close to direct port from JS to Python from the influence-utils repository.
    https://github.com/Influenceth/influence-utils/blob/00f6838b616d5c7113720b0f883c2a2d55a41267/index.js#L288

    :param a: Semi-major axis
    :param e: Eccentricity
    :param i: Inclination
    :param o: Longitude of ascending node
    :param w: Argument of periapsis
    :param m: Mean anomaly at epoch
    :param aday: Adalia day to calculate position at
    :return: xyz position
    """
    # Calculate the longitude of perihelion
    p = w + o

    # Calculate mean motion based on assumption that mass of asteroid <<< Sun
    k = 0.01720209895  # Gaussian constant (units are days and AU)
    n = k / sqrt(pow(a, 3))  # Mean motion

    # Calcualate the mean anomoly at elapsed time
    M = m + (n * aday)

    # Estimate the eccentric and true anomolies using an iterative approximation
    E = M
    last_diff = 1

    while last_diff > 0.0000001:
        E1 = M + (e * sin(E))
        last_diff = np.abs(E1 - E)
        E = E1

    # Calculate in heliocentric polar and then convert to cartesian
    v = 2 * atan(sqrt((1 + e) / (1 - e)) * tan(E / 2))
    r = a * (1 - pow(e, 2)) / (1 + e * cos(v))  # Current radius in AU

    # Cartesian coordinates
    x = (r * (cos(o) * cos(v + p - o) - (sin(o) * sin(v + p - o) * cos(i))))
    y = (r * (sin(o) * cos(v + p - o) + cos(o) * sin(v + p - o) * cos(i)))
    z = (r * sin(v + p - o) * sin(i))
    return [x * AU_MULTIPLIER, y * AU_MULTIPLIER, z * AU_MULTIPLIER]


def calculate_orbital_period(a: float):
    """
    Calculate orbital period of asteroid via keplers 3rd law.

    :param a: semi-major axis
    :return: orbital period
    """
    third_law = 0.000007495
    return int(sqrt(pow(a, 3) / third_law))


def get_current_adalia_day(display_day: bool = False):
    """
    Get the current adalia day at current time.

    :param display_day: Which timestamp to use. If true will result in the display date (such as on the website), if
    false can be used to calculate the positions in orbit
    :return: adalia day
    """
    timestamp = START_ORBIT_TIMESTAMP
    if display_day:
        timestamp = START_ARRIVAL_TIMESTAMP
    start_time = pendulum.parse(timestamp)
    current_time = pendulum.now()
    # Time diff in seconds then hours to get adalia days. 1 irl hour = 1 adalia day. Converted from seconds instead of
    # days as it's more precise.
    adalia_days = (current_time - start_time).total_seconds() / 60 / 60
    return adalia_days


def apply_position_to_df(df: pd.DataFrame):
    """
    Calculates xyz position for entire dataframe, requires orbital data as column

    :param df: dataframe to apply column to
    :return: dataframe
    """
    curr_aday = get_current_adalia_day()
    df['pos'] = [position_at_adalia_day(x['a'],
                                        x['e'],
                                        x['i'],
                                        x['o'],
                                        x['w'],
                                        x['m'],
                                        curr_aday) for x in df['orbital']]
    return df


def load_asteroids(json_file: str):
    """
    Load asteroids from the source file.

    :param json_file: json file name (path)
    :return: dataframe
    """
    json_asteroids = []
    with open(json_file) as f:
        for line in f:
            unpacked_line = ujson.loads(line)
            json_asteroids.append({'i': unpacked_line['i'],
                                   'r': unpacked_line['r'],
                                   'baseName': unpacked_line['baseName'],
                                   'orbital': unpacked_line['orbital'],
                                   'customName': unpacked_line.get('customName', '')})

    asteroids_df = pd.DataFrame(json_asteroids)  # Flatten nested JSON
    asteroids_df['orbital.T'] = [calculate_orbital_period(x['a']) for x in asteroids_df['orbital']]  # Orbital period
    asteroids_df.set_index('i', inplace=True, drop=False)  # Set asteroid ID as index

    asteroids_df = asteroids_df.astype({'i': 'int32',
                                        'r': 'int32',
                                        'orbital.T': 'int16'})  # Reduce int limit to save memory
    asteroids_df = apply_position_to_df(asteroids_df)
    return asteroids_df
