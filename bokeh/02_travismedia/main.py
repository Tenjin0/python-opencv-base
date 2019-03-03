from bokeh.plotting import figure, output_file, show, save, ColumnDataSource
import pandas
import os
df = pandas.read_csv(os.path.dirname(os.path.abspath(__file__)) + "/cars.csv")
# car = df["Car"]
# hp = df["Horsepower"]
output_file("index.html")

# Create ColumnDataSource  from data.frame

source = ColumnDataSource(df)

car_list = source.data["Car"].tolist()
p = figure(
    y_range=car_list,
    # plot_width=800,
    plot_height=400,
    x_axis_label="Horsepower",
    tools=""
)

p.hbar(
    y='Car',
    height=0.3,
    right="Horsepower",
    source=source
)

show(p)
