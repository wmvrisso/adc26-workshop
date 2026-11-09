#pragma once

namespace audio_plugin {
template <typename SampleType>
class Flanger {
public:
  void prepare(const juce::dsp::ProcessSpec& spec) {
    maxDelay_ = std::floor(static_cast<SampleType>(0.002 * spec.sampleRate));
    middleDelay_ = maxDelay_ / SampleType{2};

    reset();
  }

  template <class ProcessContext>
  void process(const ProcessContext& processContext) {
    const auto& inputBlock = processContext.getInputBlock();
    const auto& outputBlock = processContext.getOutputBlock();
    [[maybe_unused]] const auto channelCount = outputBlock.getNumChannels();
    const auto sampleCount = outputBlock.getNumSamples();

    // These sanity checks save you a headache later on in the development.
    jassert(inputBlock.getNumChannels() == channelCount);
    jassert(inputBlock.getNumSamples() == sampleCount);

    // Implementing mono first? Assert it!
    [[maybe_unused]] constexpr auto SUPPORTED_CHANNELS = 1u;
    jassert(channelCount == SUPPORTED_CHANNELS);

    // Always check if the processor is not bypassed
    if (processContext.isBypassed) {
      outputBlock.copyFrom(inputBlock);
      return;
    }

    // Process samples one by one, at least initially.
    using namespace std::views;
    constexpr auto CHANNEL = 0u;
    for (const auto sample : iota(0, static_cast<int>(sampleCount))) {
      const auto processedSample =
          processSample(inputBlock.getSample(CHANNEL, sample));
      outputBlock.setSample(CHANNEL, sample, processedSample);
    }
  }

  SampleType processSample(SampleType sample) {
    // x_h[n] = x[n] + feedback * x_h[n - D/2]
    // Delay 0 marks the last written sample, which is x_h[n - 1] at this point.
    const auto helperSample =
        sample + feedback_ * delayLine_.popSample(middleDelay_ - SampleType{1});

    // x_h[n] is written first so that the output tap can read it with delay 0.
    delayLine_.pushSample(helperSample);

    // y[n] = blend * x_h[n] + feedforward * x_h[n - D/2]
    return blend_ * helperSample +
           feedforward_ * delayLine_.popSample(middleDelay_);
  }

  void reset() { delayLine_.reset(); }

private:
  SampleType feedforward_{0.7f};
  SampleType feedback_{0.7f};
  SampleType blend_{0.7f};
  SampleType maxDelay_{0};
  SampleType middleDelay_{0};
  FractionalDelayLine delayLine_;
};
}  // namespace audio_plugin
