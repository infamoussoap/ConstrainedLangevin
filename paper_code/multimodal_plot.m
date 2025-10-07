tiledlayout(1, 2, 'TileSpacing', 'compact');

x_vals = 0:7;

ax1 = nexttile;
data = readmatrix("dikinlangevin_num_transitions.csv");
vals = 0 * x_vals;
for i = x_vals
    vals(i + 1) = sum(data == i);
end
bar(x_vals, vals);

ax1.XAxis.FontSize = 13;
ax1.YAxis.FontSize = 13;

title("Modified Dikin--Langevin", "FontSize", 20, "Interpreter", "Latex");
ylabel("Count of chains", "FontSize", 20, "Interpreter", "Latex");
xlabel("Transitions per chain", "FontSize", 20, "Interpreter", "Latex");
ylim([0, 85]);

ax2 = nexttile;
data = readmatrix("randomwalk_num_transitions.csv");
vals = 0 * x_vals;
for i = x_vals
    vals(i + 1) = sum(data == i);
end
bar(x_vals, vals);
ax2.XAxis.FontSize = 13;
title("Dikin Random Walk", "FontSize", 20, "Interpreter", "Latex");
xlabel("Transitions per chain", "FontSize", 20, "Interpreter", "Latex");
ylim([0, 50]);

linkaxes([ax1 ax2], 'y');

% Hide y tick labels on the second plot (keep ticks if you want)
set(ax2, 'YTickLabel', []);

pic_filename = 'multimodal_transition_count.pdf';
set(gcf, 'Units', 'Inches', 'Position', [1, 1, 9, 4]);
exportgraphics(gcf, pic_filename);
system("open " + pic_filename);
