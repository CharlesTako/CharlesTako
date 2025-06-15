# Final Project Setup
# Prepared by Charles Takomana, Kaushal Bhandari, and Prachi Ghoghare

library(fpp3)
library(readxl)
library(feasts)
library(fabletools)
library(imputeTS)
library(ggplot2)

# Step 1: Load & Preprocess — Monthly Granularity
df_raw <- read_excel("UNRATE.xlsx", sheet="Monthly")
glimpse(df_raw)

ggplot(df_raw, aes(x = as.Date(observation_date), y = UNRATE)) +
  geom_line(color = "black") +
  labs(title = "Original US Unemployment Data", x = "Date", y = "Unemployment Rate (%)") +
  theme_minimal()

df_ts <- df_raw %>%
  rename(Date = observation_date, UnemploymentRate = UNRATE) %>%
  mutate(Date = yearmonth(as.Date(Date))) %>%
  filter(Date >= yearmonth("2010 Jan")) %>%
  as_tsibble(index = Date)

df_ts %>% autoplot() +
  labs(title = "Monthly US Unemployment Rate (2010–2025)", y = "Rate (%)", x = "Date") +
  theme_minimal()

# Step 2: Fill monthly gaps and interpolate safely
df_ts_filled <- df_ts %>%
  fill_gaps() %>%
  mutate(UnemploymentRate = na_interpolation(UnemploymentRate))

nrow(df_ts_filled)
df_ts_filled %>% count(Date) %>% filter(n > 1)

# STL Decomposition
df_ts_filled %>%
  model(STL(UnemploymentRate)) %>%
  components() %>%
  autoplot() +
  labs(title = "STL Decomposition of US Unemployment Rate") +
  theme_minimal()

# Step 3: Stationarity Tests
df_ts_filled %>% features(UnemploymentRate, unitroot_kpss)
df_ts_filled %>% features(UnemploymentRate, unitroot_ndiffs)

# ACF & PACF of Differenced Series
df_diff <- df_ts_filled %>%
  mutate(Diff_Unemp = difference(UnemploymentRate)) %>%
  filter(!is.na(Diff_Unemp))

df_diff %>% ACF(Diff_Unemp) %>% autoplot() +
  labs(title = "ACF of Differenced Unemployment Rate") +
  theme_minimal()

df_diff %>% PACF(Diff_Unemp) %>% autoplot() +
  labs(title = "PACF of Differenced Unemployment Rate") +
  theme_minimal()

# Step 4: Data Split
split_date <- yearmonth("2023 Jan")
df_train <- df_ts_filled %>% filter(Date < split_date)
df_test <- df_ts_filled %>% filter(Date >= split_date)

# Step 5: Model Fitting
models_trained <- df_train %>%
  model(
    ETS = ETS(UnemploymentRate),
    ARIMA = ARIMA(UnemploymentRate, stepwise = FALSE, approx = FALSE),
    TSLM_Trend = TSLM(UnemploymentRate ~ trend()),
    Naive = NAIVE(UnemploymentRate),
    SNaive = SNAIVE(UnemploymentRate),
    Mean = MEAN(UnemploymentRate),
    RWDrift = RW(UnemploymentRate ~ drift())
  )

# Step 6: Manual Model Additions
models_manual <- df_train %>%
  model(
    ETS_AAN = ETS(UnemploymentRate ~ error("A") + trend("A") + season("N")),
    ETS_MAM = ETS(UnemploymentRate ~ error("M") + trend("A") + season("M")),
    ARIMA010 = ARIMA(UnemploymentRate ~ pdq(0,1,0)),
    TSLM_Trend_Season = TSLM(UnemploymentRate ~ trend() + season())
  )

# Forecasting
fc_all <- bind_rows(
  forecast(models_trained, h = nrow(df_test)),
  forecast(models_manual, h = nrow(df_test))
)

# Accuracy Evaluation
accuracy_tbl <- fc_all %>% accuracy(df_test)
accuracy_tbl %>% select(.model, RMSE, MAE, MAPE) %>% arrange(RMSE)

# Step 7: Residual Diagnostics
aug_trained <- models_trained %>% augment()
aug_manual <- models_manual %>% augment()
aug_all <- bind_rows(aug_trained, aug_manual)

resid_diag <- aug_all %>%
  features(.resid, ljung_box, lag = 12)

print("Ljung-Box Test Results:")
print(resid_diag)

# ACF Comparison of Residuals
aug_all %>%
  group_by(.model) %>%
  ACF(.resid) %>%
  filter(lag <= 20) %>%
  ggplot(aes(x = lag, y = acf, fill = .model)) +
  geom_col(position = "dodge", width = 0.6) +
  geom_hline(yintercept = c(-0.2, 0.2), linetype = "dashed", color = "blue") +
  labs(title = "ACF Comparison of Residuals", x = "Lag", y = "ACF") +
  theme_minimal() +
  theme(legend.title = element_blank())

# Forecast Visualization
fc_all %>%
  autoplot(df_ts_filled, level = 95) +
  labs(title = "Forecasts of U.S. Unemployment Rate", y = "Unemployment Rate (%)", x = "Date") +
  facet_wrap(~.model, scales = "free_y") +
  theme_minimal()

# Focus Plot for Best Models: ARIMA and SNaive
fc_all %>%
  filter(.model %in% c("ARIMA", "SNaive")) %>%
  autoplot(df_ts_filled, level = 95, size = 1.2) +
  labs(title = "Top Model Forecasts: ARIMA and SNaive",
       subtitle = "Comparison of the two best-performing models",
       y = "Unemployment Rate (%)", x = "Date") +
  facet_wrap(~.model, scales = "free_y") +
  theme_minimal()
