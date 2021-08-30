#Anomaly Building Blocks

## Implemented Functions

The implemented functions of anomalies in the old AnomalyGenerator are:

1. Flatline
   - The sensor gets suck at the value at time $t$ for a random/short timespan
2. Zero-Value
   - Sensor value drops to $0$ at time $t$ for a random/short timespan
3. Outlier
   - Sensor value spikes up (or down) for a single timestamp
4. Increased Noise
   - Whitenoise in the signal is increased  for a random/short timespan
5. Change Point
   - Signal "baseline" jumps up or down to a new level
6. Min/Max-Clipping
   - Values at the maximum or minimumum (or both) are clipped to a new value resulting in a flatline
7. Additional Trend
   - Values are increasing (or decreasing) in a unwanted way
8. Electric Feedback
   - The sensor has a malfunction and measures the electrical current (50Hz or 60Hz) in addition to the normal measurement.
9. Maximum Smoothing
   - The maxima and minima are smoothend by the moving average
10. Ratio Compression
    - The maxima and minima are compressed by a userdefined ratio

###Basic Building Blocks

Here we try to make a list of the basic building blocks that cover all the function mentioned above. There are two classes of functions to manipulate the signal. 

1. Detection Functions
2. Manipulation Functions
3. Value Creation Functions



##### Detection Funtions

Detection functions are tools to get certain parts of a time series. E.g.:

- Based on values of the time series (i.e. $>,<, =, etc.$) and any combination of these operators
- Based on the time index in the time series (most of these are implemented already in the time series)



##### Manipulation function

Functions that manipulte the values of the time series

- addition -> adding given value to the series (also subtration)
- Multiply -> multiply given value to the series (also division)
- replace -> replace the values in the series
- Expand -> adding the values between two timestamps and correcting the timestamps
- crop -> removing values from the series (start and end is implemented -> middle not)



##### Value Functions

- Outlier

  - Change Point
  - Outlier

- Time $t$ value

  - flatline
  - Min/Max-Clipping

- Electric Feedback

- Noise

  - increaded noise

- Trend

- Moving Average

  





