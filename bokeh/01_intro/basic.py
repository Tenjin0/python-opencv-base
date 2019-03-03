from bokeh.plotting import figure, output_file, show


x = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
y = [i + 2 for i in x]

output_file("lines.html")

p = figure(
    tools="pan,box_zoom,reset,save",
    title="axis example",
    x_range=(0, 3),
    y_range=[0, 20],
    x_axis_label="sections",
    y_axis_label="particles"
)
p.line(x, y, legend="y=x")

show(p)
