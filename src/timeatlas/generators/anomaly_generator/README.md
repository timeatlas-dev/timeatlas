##### flatline: 

The value of $t_{-1}$ is repeated for $\delta{t}$ steps

Parameters:
 - data: TimeSeries object
 - n: number of anomalies per TimeSereies object
 - minimum: minimum length of the anomaly
 - maximum: maximum length of the anomaly
 

##### zeroing: 

The value is droping to the default value of the sensor (zero or NaN)

Parameters:
 - data: TimeSeries object
 - n: number of anomalies per TimeSereies object
 - value: the value inserted to the timesereis (0 or NaN)
 - minimum: minimum length of the anomaly
 - maximum: maximum length of the anomaly
 
 
##### outlier: 

A sudden spike in the TimeSeries

Parameters:
 - data: TimeSeries object
 - n: number of anomalies per TimeSereies object
 - sigma: sigma of the normal distributed with value added to the datapoint $d_t$ with $\mu d_t$
 
##### increase_noise: 

A sudden spike in the TimeSeries

Parameters:
 - data: TimeSeries object
 - n: number of anomalies per TimeSereies object
 - sigma: sigma of the normal distributed with value added to the datapoint $d_t$ with $\mu d_t$
 
##### change_point: 

A sudden spike in the TimeSeries where the data stays for a time $\delta t$.
This creates a flatline.

Parameters:
 - data: TimeSeries object
 - n: number of anomalies per TimeSereies object
 - ending: if `True` the data will change back else it will stay until the end of the TimeSeries
 - change_point_factor: sigma of the normal distributed change point
 - minimum: minimum length of the anomaly
 - maximum: maximum length of the anomaly
 
##### clipping: 

The TimeSeries values are clipped at a given threshold

Parameters:
 - data: TimeSeries object
 - n: number of anomalies per TimeSereies object
 - clip_value: value threshold at which to clip
 - mode ('top', 'bottom' or 'both'): clipping happens in positiv, negative and both
 
##### trend: 

Introducing a trend into the TimeSeries

Parameters:
 - data: TimeSeries object
 - n: number of anomalies per TimeSereies object
 - slope: the slope m of the trends (y = x*m)
 
##### electric_feedback: 

Introducing an electric feedback into the TimeSeries using a Fast Fourier Transform.

Parameters:
 - data: TimeSeries object
 - sampling_speed: frequency of the sampling (number of seconds between two indices)
 - amp: amplitude of the feedback 
 - hrz: hertz of the frequency introduced
 
##### hard_knee: 

Squeeze the values above the threshold by a factor (0 < factor <= 1)

Parameters:
 - data: TimeSeries object
 - n: number of anomalies per TimeSereies object
 - threshold: value where the factor is applied
 - factor (< 1): factor how hard the squeeze is
 
##### max_smoothing: 

values above threshold will be smoothed by the moving average.

Parameters:
 - data: TimeSeries object
 - n: number of anomalies per TimeSereies object
 - window_size: size of the window over which the moving average is calculated
 - threshold: theshold above the avarage is introduced
 
 
##### ratio_compression: 

Values above a threshold will be compressed by a ratio. Eg. 1:4 -> the top quarter, above the threshold, will be removed

Parameters:
 - data: TimeSeries object
 - n: number of anomalies per TimeSereies object
 - ratio: ratio of the compression
 - threshold: theshold above the compression is applied