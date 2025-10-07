dt_vals = [0.05, 0.01, 0.005, 0.001];
expected = 0.950428661907089;

for i = 1:4
    dt = dt_vals(i);
    filename = "traj_dt_" + string(dt) + ".csv";
    data = readmatrix(filename);
    norm = vecnorm(data, 2, 2);

    cum_mean = cumsum(norm) ./ (1:length(norm)).';
    
    semilogy((1:length(cum_mean)) * 0.1, abs(cum_mean - expected), 'DisplayName', "$\textrm{dt} = " + string(dt) + "$", "linewidth", 4);
    hold on;
end
ax = gca;

legend("Interpreter", "Latex", "FontSize", 13, 'location', 'best');
xlim([0, 5000]);
xticks(0:1000:5000)
set(gca, 'FontSize', 13);
% ylabel("$\bigg{|}{E}[\|x\|]-\frac{1}{N}\sum_{n=1}^N\|X_{n}\|\bigg{|}$", "Interpreter", "Latex", "FontSize", 15)
ylabel("Error", "Interpreter", "Latex", "FontSize", 20);
xlabel("Integration Time", "Interpreter", "Latex", "FontSize", 20)
title("Discretization Bias of the Dikin-Langevin SDE", "Interpreter", "Latex", "FontSize", 20)

pic_filename = 'Gaussian_samples.pdf';
set(gcf, 'Units', 'Inches', 'Position', [1, 1, 8, 4]);
exportgraphics(gcf, pic_filename);
system("open " + pic_filename);
