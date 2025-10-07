filenames = ["dikinlanvegin_cummean_norm.csv","randomwalk_cummean_norm.csv","langevin_cummean_norm.csv"];
sampler_names = ["Modified Dikin--Langevin","Dikin Random Walk","MALA"];
expected_value = 0.4446709099274374;

tiledlayout(1,3,'Padding','tight','TileSpacing','tight');
ax = gobjects(1,3);               % collect axes

step = 200;
for i = 1:3
    ax(i) = nexttile; hold(ax(i),'on');
    set(ax(i),'YScale','log');

    data = readmatrix(filenames(i));
    err_runs = abs(data(1:step:end, :) - expected_value).';   % R x T

    q = quantile(err_runs, [0.1 0.5 0.9], 1);  % along runs
    lo  = squeeze(q(1,:,:));                   % 1 x T
    med = squeeze(q(2,:,:));
    hi  = squeeze(q(3,:,:));

    % clamp lower ribbon edge for log-y (avoid zeros)
    lo = max(lo, eps);

    if i == 1
        ax(i).YAxis.FontSize = 12;    
    else
        ax(i).YTickLabel = [];       % no tick labels
    end

    x_vals = 1:numel(med);
    patch(ax(i), [x_vals fliplr(x_vals)], [hi fliplr(lo)], 'red', ...
          'FaceAlpha', 0.5, 'EdgeColor','none');
    plot(ax(i), x_vals, med, 'k', 'LineWidth', 2);

    T = length(data); 

    quarters_true = round(T * [0.50 1.00]);       % true iters
    tick_pos = 1 + floor((quarters_true - 1) / step);       % positions in subsampled x
    % tick_pos(tick_pos > numel(x_vals)) = numel(x_vals);     % guard
    % x_tick_labels = ["$25\,000$", "$50\,000$", "$75\,000$", "$100\,000$"];
    x_tick_labels = ["50 000", "100 000"];
    set(ax(i), 'XTick', tick_pos, ...
               'XTickLabel', x_tick_labels, ... % add thousands separator
               'fontsize', 12);
               % 'TickLabelInterpreter','latex', 'fontsize', 12);
    xlim([0, length(med)]);

    xlabel(ax(i), 'Iteration', 'FontSize', 18, 'Interpreter','latex');
    title(ax(i), sampler_names(i), 'Interpreter','latex', 'FontSize', 18);
end

% make all tiles share the same y-axis and set limits once
linkaxes(ax, 'y');
ylim(ax(1), [1e-3 0.45]);         % pick whatever common range you want

% one shared y-label (preferred over repeating in the loop)
ylabel(ax(1), '$\Big{|}{\|x\|}^2_{1:t}-\mu^\ast\Big{|}$', ...
       'Interpreter','latex','FontSize',18);

set(gcf, 'Units','inches', 'Position',[1 1 10 3.5]);
exportgraphics(gcf, 'metropolis_results.pdf');
system("open metropolis_results.pdf");
