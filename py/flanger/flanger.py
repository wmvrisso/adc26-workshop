import numpy as np


def clp2(x):
    return 1 << (x - 1).bit_length() if x > 0 else 0


class DelayLine:
    """
    Implements a delay line with integer sample delay, used for storing and
    reading delayed audio samples.

    Attributes:
        wrap_mask (int): Mask used for wrapping buffer indices, set to size - 1.
        buffer (numpy.ndarray): A buffer array to hold the delayed samples.
        write_pointer (int): The current index for writing in the buffer.
    """

    def __init__(self, size: int):
        """
        Initializes the delay line buffer with a specified size. The size is
        adjusted to the next power of 2 for efficient wrapping.

        Parameters:
            size (int): The size of the delay line buffer.
        """
        size = clp2(size)
        self.wrap_mask = size - 1
        self.buffer = np.zeros((size,))
        self.write_pointer = 0

    def push_sample(self, sample: float):
        """
        Writes a sample to the delay line buffer at the current write pointer
        location and advances the write pointer with wrapping.

        Parameters:
            sample (float): The sample value to write into the buffer.
        """
        self.buffer[self.write_pointer] = sample
        self.write_pointer = (self.write_pointer + 1) & self.wrap_mask

    def pop_sample(self, delay: int):
        """
        Reads a delayed sample from the buffer based on a specified integer
        delay.

        Parameters:
            delay (int): The delay in samples to read from the buffer.

        Returns:
            float: The delayed sample value.

        Raises:
            IndexError: If the delay is negative or exceeds the buffer size.
        """
        if delay < 0:
            raise IndexError("delay smaller than 0")

        if self.buffer.size <= delay:
            raise IndexError("delay larger than buffer indices")

        # The write pointer always advances after write.
        # Thus, we must use -1 to have delay 0 mark the last written sample.
        read_pointer = (self.write_pointer - 1 - delay) & self.wrap_mask
        return self.buffer[read_pointer]

    def print_buffer(self):
        """
        Prints the current state of the delay line buffer.
        """
        print(self.buffer)


class FractionalDelayLine(DelayLine):
    """
    Extends DelayLine to support fractional sample delay by interpolating between
    two samples, providing non-integer delays.
    """

    def __init__(self, size: int):
        super().__init__(size)

    def pop_sample(self, delay: float):
        """
        Reads a sample from the delay line buffer using fractional delay. Uses
        linear interpolation between two samples for smoother delay response.

        Parameters:
            delay (float): The fractional delay in samples.

        Returns:
            float: The delayed sample value using fractional delay.

        Raises:
            IndexError: If the integer part of delay is out of bounds.
        """
        int_delay = int(delay)

        if int_delay == delay:
            return super().pop_sample(int_delay)

        frac_delay = delay - int_delay

        sample_1 = super().pop_sample(int_delay)
        sample_2 = super().pop_sample(int_delay + 1)

        return frac_delay * sample_2 + (1 - frac_delay) * sample_1


class LFO:
    def __init__(self, sample_rate) -> None:
        self.phase = 0
        self.phase_increment = 0
        self.sample_rate = sample_rate

    def set_frequency(self, frequency_hz):
        self.phase_increment = 2 * np.pi * frequency_hz / self.sample_rate

    def get_next_value(self):
        """
        Generates next LFO output sample. This advances internal phase
        so 2 subsequent calls to this method will result in different values.

        Returns:
            float: the generated sine sample
        """
        value = np.sin(self.phase)
        self.phase += self.phase_increment
        self.phase = np.fmod(self.phase, 2 * np.pi)
        return value


class Flanger:
    def __init__(self, sample_rate):
        self.max_delay = int(0.002 * sample_rate)
        # Delays up to and including max_delay must be readable, hence the + 1.
        self.delay_line = FractionalDelayLine(self.max_delay + 1)
        self.middle_delay = self.max_delay / 2
        self.feedback = 0.7
        self.feedforward = 0.7
        self.blend = 0.7
        self.lfo = LFO(sample_rate)
        self.lfo.set_frequency(2)

    def process_sample(self, input_sample):
        # m = s_LFO,unipolar[n] * D
        lfo_unipolar = 0.5 * (self.lfo.get_next_value() + 1)
        modulated_delay = lfo_unipolar * self.max_delay

        # x_h[n] = x[n] + feedback * x_h[n - D/2]
        # Delay 0 marks the last written sample, which is x_h[n - 1] at this point.
        x = input_sample
        x_h = x + self.feedback * self.delay_line.pop_sample(self.middle_delay - 1)

        # x_h[n] is written first so that m = 0 can read it with delay 0.
        self.delay_line.push_sample(x_h)

        # y[n] = blend * x_h[n] + feedforward * x_h[n - m]
        return self.blend * x_h + self.feedforward * self.delay_line.pop_sample(
            modulated_delay
        )


def apply_flanger(samples, sample_rate):
    output = np.zeros_like(samples)
    flanger = Flanger(sample_rate)

    for i in range(samples.shape[0]):
        output[i] = flanger.process_sample(samples[i])

    return output
