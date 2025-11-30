import pandas as pd
from dfcleaner import DFCleaner  # Makes data cleaning easier/quicker
from chartengineer import ChartMaker
import os

cleaner = DFCleaner()
chart_maker = ChartMaker()

raw_df = cleaner.to_df(r'E:\Projects\MacroCrypto\tests\backtest_results.csv') # get from the api instead

print("Initial DataFrame:")
print(raw_df.head())
print(raw_df.tail())
print(raw_df.info())
print(raw_df.describe())

df, time_freq =  cleaner.to_time(raw_df)

chart_maker.build(
    df = df,
    axes_data=dict(y1=df.columns.to_list()),
    title="Backtest Results Over Time",
    chart_type={'y1':'line'},
    options=dict(show_legend=True, ticksuffix=dict(y1='%'))
)

chart_maker.show_fig(browser=True)
