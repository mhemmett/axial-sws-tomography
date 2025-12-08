import pandas as pd
from obspy.clients.fdsn import Client
from obspy.core.utcdatetime import UTCDateTime
from obspy import Stream

def get_all_traces(df, filename, starttime, endtime):
    """
    Loops over a DataFrame of earthquakes with 't_start' and 't_final' columns.
    Retrieves waveform data from IRIS and saves all traces into one MiniSEED file.
    """
    client = Client("IRIS")

    # Define unique station-channel combinations
    combinations = [
        ('AXCC1', 'HHE'), ('AXCC1', 'HHN'), ('AXCC1', 'HHZ'), ('AXCC1', 'HDH'),
        ('AXEC1', 'EHE'), ('AXEC1', 'EHN'), ('AXEC1', 'EHZ'),
        ('AXEC2', 'HHE'), ('AXEC2', 'HHN'), ('AXEC2', 'HHZ'), ('AXEC2', 'HDH'),
        ('AXEC3', 'EHE'), ('AXEC3', 'EHN'), ('AXEC3', 'EHZ'),
        ('AXAS1', 'EHE'), ('AXAS1', 'EHN'), ('AXAS1', 'EHZ'),
        ('AXAS2', 'EHE'), ('AXAS2', 'EHN'), ('AXAS2', 'EHZ'),
        ('AXID1', 'EHE'), ('AXID1', 'EHN'), ('AXID1', 'EHZ'),
        ('AXBA1', 'HHE'), ('AXBA1', 'HHN'), ('AXBA1', 'HHZ'), ('AXBA1', 'HDH')
    ]

    all_traces = Stream()

    for _, row in df.iterrows():
        t_start = UTCDateTime(row[str(starttime)])
        t_final = UTCDateTime(row[str(endtime)])

        for station, channel in combinations:
            try:
                stream = client.get_waveforms(
                    network='OO',
                    station=station,
                    location='',
                    channel=channel,
                    starttime=t_start,
                    endtime=t_final,
                    attach_response=True
                )
                all_traces += stream
                print(f"Retrieved data for {station} {channel} from {t_start} to {t_final}")
            except Exception as e:
                print(f"No data for {station} {channel} - {e}")

    if all_traces:
        all_traces.write(str(filename) + ".mseed", format="MSEED")
        print(f"Saved all waveform data to {filename}.mseed")

    return all_traces

def get_station_traces(df, filename, starttime, endtime, station_id):
    """
    Loops over a DataFrame of earthquakes with 't_start' and 't_final' columns.
    Retrieves waveform data from IRIS and saves all traces into one MiniSEED file.
    """
    client = Client("IRIS")

    # Define unique station-channel combinations
    combinations = [
        ('AXCC1', 'HHE'), ('AXCC1', 'HHN'), ('AXCC1', 'HHZ'),
        ('AXEC1', 'EHE'), ('AXEC1', 'EHN'), ('AXEC1', 'EHZ'),
        ('AXEC2', 'HHE'), ('AXEC2', 'HHN'), ('AXEC2', 'HHZ'),
        ('AXEC3', 'EHE'), ('AXEC3', 'EHN'), ('AXEC3', 'EHZ'),
        ('AXAS1', 'EHE'), ('AXAS1', 'EHN'), ('AXAS1', 'EHZ'),
        ('AXAS2', 'EHE'), ('AXAS2', 'EHN'), ('AXAS2', 'EHZ'),
        ('AXID1', 'EHE'), ('AXID1', 'EHN'), ('AXID1', 'EHZ'),
    ]

    all_traces = Stream()

    for _, row in df.iterrows():
        t_start = UTCDateTime(row[str(starttime)])
        t_final = UTCDateTime(row[str(endtime)])
        current_station = row[str(station_id)]

        for station, channel in combinations:
            if station == current_station:
                try:
                    stream = client.get_waveforms(
                        network='OO',
                        station=station,
                        location='',
                        channel=channel,
                        starttime=t_start,
                        endtime=t_final,
                        attach_response=True
                    )
                    all_traces += stream
                    print(f"Retrieved data for {station} {channel} from {t_start} to {t_final}")
                except Exception as e:
                    print(f"No data for {station} {channel} - {e}")
            else:
                continue

    if all_traces:
        all_traces.write(str(filename) + ".mseed", format="MSEED")
        print(f"Saved all waveform data to {filename}.mseed")

    return all_traces