# Bandpower estimation

The plot simulates an online power estimation with various methods. For this purpose,
the raw signal (sampling frequ. $f_s=256\,\text{Hz}$) is considered to be incoming sample by sample. All methods for real time power estimation come with pros and cons. Methods are still being actively researched ([Busch, et. al., 2021](https://www.sciencedirect.com/science/article/pii/S0014488621002776?via%3Dihub), [Schreglmann, et. al., 2021](https://www.nature.com/articles/s41467-020-20581-7)). This plot is only ment to provide a qualitative overview of the sequential processing of a few selected methods without claimim completeness or selecting the most accurate models.

### Periodogram approach

The periodogram approach is a common method to estimate the power spectral density (PSD) of a signal. It is based on the Fourier transform of the signal. To simulate the effect as if unlimited processing power was available, we calculate the power estimates over a window of `Rolling window length` seconds, for which periodogram's can be calculated and reasonably be averaged, enabling e.g., the use of Welch's method, which is commonly used ([Tepper, et. al., 20217](https://onlinelibrary.wiley.com/doi/10.1155/2017/1512504)). The delayed response of methods which compute over longer windows can be modulated via `Rolling window length` parameter. Analogeously, the [multitaper method from mne](https://mne.tools/stable/generated/mne.time_frequency.psd_array_multitaper.html) is used to estimate the PSD. The bandpower is the provided as average power over all frequency bins for a given frequency range.

### Hilbert approach

The [Hilbert transformation](https://de.wikipedia.org/wiki/Hilbert-Transformation) approach could be used to estimate the bandpower of a narrow band signal. More accurately, an endpoint corrected version of the Hilbert transform should be used for real time computation [Schreglmann, et. al., 2021](https://www.nature.com/articles/s41467-020-20581-7). We stick to the standard Hilbert transform provided by [`scipy.signal.hilbert`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html) for simplicity sake.

### Rectification approach

Another approach common, e.g., in the context of adaptive deep brain stimulation, is to calculate power estimates as the mean of a rectified signal ([Tinkhauser, et. al., 2017](https://academic.oup.com/brain/article/140/4/1053/2993763)).

### Literature

- [Real-time phase and amplitude estimation of neurophysiological signals exploiting a non-resonant oscillator. Busch, et. al., 2021](https://www.sciencedirect.com/science/article/pii/S0014488621002776?via%3Dihub)

- [Non-invasive suppression of essential tremor via phase-locked disruption of its temporal coherence. Schreglmann, et. al., 2021](https://www.nature.com/articles/s41467-020-20581-7)

- [Selection of the Optimal Algorithm for Real-Time Estimation of Beta Band Power during DBS Surgeries in Patients with Parkinson’s Disease. Tepper, et. al., 2017](https://onlinelibrary.wiley.com/doi/10.1155/2017/1512504)

- [The modulatory effect of adaptive deep brain stimulation on beta bursts in Parkinson’s disease. Tinkhauser, et. al., 2017](https://academic.oup.com/brain/article/140/4/1053/2993763)
