import csv
import sys
from dataclasses import dataclass
import numpy as np


def csv_read_matrix(file_path, delim=',', comment_str="#"):
    """Parse a csv-like file into a matrix

    Args:
        file_path: A file handle or path to a csv file
        delim: Symbol for the delimeter
        comment_str: Symbol for comments. Lines that start with this symbol are ignored.

    Returns: 2D list. Each row corresponds to a line in the file. Every element is stored as a string.
    """
    file_handle = file_path

    if not hasattr(file_path, 'read'):
        file_handle = open(file_path)

    generator = (line for line in file_handle if not line.startswith(comment_str))
    reader = csv.reader(generator, delimiter=delim)
    data = [row for row in reader]
    return data


@dataclass
class TimeStampStreamInfo():
    # Stream of timestamps.
    timestamp_stream: np.ndarray
    # The current index within the stream.
    cur_index: int = 0
    # We may modify the size of the 'timestamp_stream'. Here we store the necessary offset to map internal indices
    # back to its original size.
    offset: int = 0


class TimestampIndex():
    def __init__(self,timestamp,index):
        self.timestamp = timestamp
        self.index = index


class TimestampSynchronizer():
    """Class used to synchronize various datastreams with timestamps.

    """
    def __init__(self, max_allowed_dt_between_timestamps=int(5000192 / 2)):

        self.timestamps_dict = {}

        self.min_timestamp = float('inf')
        self.max_timestamp = float('-inf')

        self.start_timestamp = None
        self.end_timestamp = None

        self.current_timestamp = None

        # Set the maximum dt allowed to consider 2 timestamps to be equal.
        self.max_dt_between_timestamps = max_allowed_dt_between_timestamps

    def add_timestamp_stream(self, data_name, timestamps):
        """Add a stream of timestamps for the class to keep track of.

        Args:
            data_name: Name or ID of the stream of timestamps. Usually the sensor name.
            timestamps: Sorted List of timestamps.

        """

        if data_name in self.timestamps_dict:
            logger.error("Data Stream with name %s already exist", data_name)

        # Convert to numpy to make processing faster
        self.timestamps_dict[data_name] = TimeStampStreamInfo(np.array(timestamps), 0)
        self.current_timestamp = timestamps[0]
        if timestamps[0] < self.min_timestamp:
            self.min_timestamp = timestamps[0]
        if timestamps[-1] > self.max_timestamp:
            self.max_timestamp = timestamps[-1]

    def set_start_timestamp(self, timestamp):
        """Set the timestamp from which we want to start our data processing.

        Args:
            timestamp:

        """
        self.start_timestamp = timestamp
        self.current_timestamp = timestamp

        for value in self.timestamps_dict.values():
            stream = value.timestamp_stream
            size = stream.size
            value.timestamp_stream = stream[stream >= timestamp - self.max_dt_between_timestamps]
            value.offset = size - value.timestamp_stream.size

    def set_end_timestamp(self, timestamp):
        self.end_timestamp = timestamp

        for value in self.timestamps_dict.values():
            stream = value.timestamp_stream
            value.timestamp_stream = stream[stream <= timestamp]

    def get_current_minimum_timestamp(self):
        """Check our various timestamp streams and return whichever has the lowest value.

        Returns:


        """
        min_timestamp = sys.float_info.max
        for timestamp_index in self.timestamps_dict.values():
            if timestamp_index.cur_index == timestamp_index.timestamp_stream.size - 1:
                continue
            cur_timestamp = timestamp_index.timestamp_stream[timestamp_index.cur_index]
            if cur_timestamp <= min_timestamp:
                min_timestamp = cur_timestamp
        return min_timestamp

    def get_data(self):
        """Get the data at our current timestamp.

        Returns:
                A dictionary which maps sensor name to a timestamp and index. The sensor is only included if it has
                data at the current timestamp.
        """
        min_timestamp = self.get_current_minimum_timestamp()
        self.current_timestamp = min_timestamp
        output_data = {}
        for data_name, value in self.timestamps_dict.items():
            cur_index = value.cur_index
            stream = value.timestamp_stream
            # Check datastream is not at the end
            if cur_index == stream.size - 1:
                continue
            timestamp = stream[cur_index]
            if abs(timestamp - min_timestamp) <= self.max_dt_between_timestamps:
                output_data[data_name] = TimestampIndex(timestamp,cur_index + value.offset)
                value.cur_index += 1
        return output_data

    def has_data(self):
        if self.current_timestamp >= self.max_timestamp:
            return False
        return True
