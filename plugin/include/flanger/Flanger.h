#pragma once

namespace audio_plugin {
template <typename SampleType>
class Flanger {
public:
  struct Parameters {
    using Ptr = std::shared_ptr<Parameters>;

    wolfsound::Frequency lfoFrequency = wolfsound::Frequency{2.f};
  };

  Flanger() = default;

  explicit Flanger(typename Parameters::Ptr parameters)
      : parameters_{std::move(parameters)} {}

  void setParameters(const Parameters& parameters) {
    lfo_.setFrequency(parameters.lfoFrequency.value());
  }

  void prepare(const juce::dsp::ProcessSpec& spec) {
    constexpr auto MAX_DELAY_SECONDS = 0.002;
    maxDelay_ =
        static_cast<SampleType>(std::ceil(spec.sampleRate * MAX_DELAY_SECONDS));
    middleDelay_ = maxDelay_ / SampleType(2);

    lfo_.prepare(spec);
    reset();
    lfo_.setFrequency(parameters_->lfoFrequency.value(), true);
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

    setParameters(*parameters_);

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
    const auto& x = sample;
    const auto xh = x + feedback_ * delayLine_.popSample(middleDelay_);

    // juce::dsp::Oscillator adds the generated sample to its input.
    const auto lfoValue = lfo_.processSample(SampleType(0));
    const auto lfoUnipolar = SampleType(0.5) * (lfoValue + SampleType(1));
    const auto currentDelay = lfoUnipolar * maxDelay_;

    const auto y =
        blend_ * xh + feedforward_ * delayLine_.popSample(currentDelay);

    delayLine_.pushSample(xh);

    return y;
  }

  void reset() {
    delayLine_.reset();
    lfo_.reset();
  }

private:
  SampleType feedforward_ = SampleType(0.7);
  SampleType feedback_ = SampleType(0.7);
  SampleType blend_ = SampleType(0.7);
  FractionalDelayLine delayLine_;
  SampleType maxDelay_{};
  SampleType middleDelay_{};
  juce::dsp::Oscillator<SampleType> lfo_{
      [](SampleType x) { return std::sin(x); }};
  typename Parameters::Ptr parameters_ = std::make_shared<Parameters>();
};
}  // namespace audio_plugin
