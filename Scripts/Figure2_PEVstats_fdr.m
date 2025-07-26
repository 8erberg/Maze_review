function Figure2_PEVstats_fdr

directory = 'E:\Lab\BlackRock\data\Plots and data\PEV\'; 

 %% Average PEV Goal(TGdelay)&Move(ALTon1 HvsV)
    %%%%%%%%%%%%%%
    %load data 
    GxM = load([directory,'GxM_FDR.mat']); %loads structure with real and randomised PEV results for each neuron in each epoch
    results = load([directory,'results_FDR.mat']); %loads the structure with FDR adjusted p_valuesXepoch for areas comparisons
    %define epochs, times and areas
    binsize = 0.05;
    epochs =  {'TG_delay','ALT1_on','ALT2_on'};
    zeros =  {'Goal onset','Choice onset','Choice onset'};
    etitle = {'Goal','Step 1 (HvsV)', 'Step 2 (2vs4)'};
    TGepoch = [-0.2, 0.9];
    Altepoch = [-0.2, 0.4];
    time_ticks.TG_delay = [TGepoch(1):binsize/2:TGepoch(end)];
    time_ticks.ALT1_on = [Altepoch(1):binsize/2:Altepoch(end)];
    time_ticks.ALT2_on = [Altepoch(1):binsize/2:Altepoch(end)];
    areas = unique(GxM.info(:,2));
    areas_sel = {'Insula';'dlPM';'dmPFC';'vPFC'};

    colours1 = ["#e37a7a ";"#70a44c";"#6091d6";"#ffc000";"#FADCC8";"#FF7F50"];%valentina3

    areas_sel = {'vPFC';'dlPM';'dmPFC';'Insula'};
    areas={'vPFC';'dlPM';'dmPFC';'Insula';'PS';'dPFC'};
    areas_legend={'vlPFC';'dPM';'dmPFC';'I/O'};
    results.label=[results.label(5);results.label(6);results.label(3);results.label(4);results.label(1);results.label(2)];
    for x=1:length(epochs)
        results.(epochs{x})=[results.(epochs{x})(5,:);results.(epochs{x})(6,:);results.(epochs{x})(3,:);results.(epochs{x})(4,:);results.(epochs{x})(1,:);results.(epochs{x})(2,:)];
    end
    
    %comparison lines parameters
    colours1_sel = colours1(ismember(areas,areas_sel),:);
    shape_mrk = 'o'; size_mrk = 4;
    %for plotting significance lines under the areas plot
    all_comb = nchoosek(1:size(areas_sel,1),2);
    start = [-0.0025:-0.001:-0.0075]; 
    step = -0.00025; %this has to change when changing LineWidth
    %%%%%%%%%%%%%%

    figure

    for e =1:length(epochs)
        RomegaXarea = [];
        epoch = epochs{e};

        sign = double(results.(epoch)>=0.05); 
        sign(sign==1) = NaN;

        subplot(1,length(epochs),e)
        for a = 1:length(areas)
            area = areas{a};
            idXarea =  ismember(GxM.info(:,2),areas{a}); %finding neuron_ids of a certain area for both monkeys
            numXarea = sum(idXarea,1);
            omegaXarea =  mean(GxM.(epoch).Real_PEV(idXarea,1:end-2),1,'omitnan');
            for r = 1:1000
                RomegaXarea(r,:) = mean(GxM.(epoch).Rand_PEV(idXarea,1:end-2,r),1,'omitnan');
            end
            %Do FDR
            proportionsXarea = sum(RomegaXarea >= omegaXarea,1)./size(RomegaXarea,1);
            [h,~, ~, adj_p] = fdr_bh(proportionsXarea,0.05,'pdep','no');

            hold on
            %plot PEV values for 4 areas only
            if a<=length(areas_sel) 
                plot(time_ticks.(epoch)(2:2:end),omegaXarea,'-','color',colours1(a,:),'LineWidth',2, 'Marker',shape_mrk,'MarkerSize',size_mrk,'MarkerFaceColor',colours1(a,:),'MarkerEdgeColor',colours1(a,:));
                Grey_NaN = NaN(1,size(omegaXarea,2));
                Grey_NaN(h == 0) = omegaXarea(h == 0);
                plot(time_ticks.(epoch)(2:2:end), Grey_NaN ,'color',colours1(a,:),'LineWidth',2, 'Marker','square','MarkerSize',8,'MarkerFaceColor',colours1(a,:),'MarkerEdgeColor',colours1(a,:),'HandleVisibility','off'); % plot only not significative data (=0) with grey line
            end

            % Parameters to plot significance lines under the main plot          
            % Calculate the actual bin edges (start and end times for each bin)
            bin_width = 0.05;
            x = time_ticks.(epoch)(2:2:end);
            x_start = x - bin_width/2;  % Start times of bins
            x_end = x + bin_width/2;    % End times of bins

            ylim([-0.01 0.018])
            pair1 = sign(a,:) + start(a) - 0.001;
            pair2 = sign(a,:) + start(a) - 0.001 + step;
            
            % Create vectors for plotting horizontal lines
            x_plot = [];
            y_plot_pair1 = [];

            % For each non-NaN value, create the horizontal line segments
            for i = 1:length(pair1)
                if ~isnan(pair1(i))
                    % Add points for this bin (including NaN to break the line if needed)
                    x_plot = [x_plot, x_start(i), x_end(i), NaN];
                    y_plot_pair1 = [y_plot_pair1, pair1(i), pair1(i), NaN];
                end
            end
            %plot significance lines
            y_plot_pair2 = y_plot_pair1 + step;
            plot(x_plot,y_plot_pair1,'-','color',colours1_sel(all_comb(a,1),:),'LineWidth',4,'HandleVisibility','off');
            hold on
            plot(x_plot,y_plot_pair2,'-','color',colours1_sel(all_comb(a,2),:),'LineWidth',4,'HandleVisibility','off');

        end %end areas

        xline(0,'-',zeros{e},'HandleVisibility','off');
        ylabel('Average PEV')
        xlabel('Time (s)')
        title(etitle{e},'Interpreter', 'none')
        legend(areas_legend)

    end %end epochs


end

% fdr_bh() - Executes the Benjamini & Hochberg (1995) and the Benjamini &
%            Yekutieli (2001) procedure for controlling the false discovery 
%            rate (FDR) of a family of hypothesis tests. FDR is the expected
%            proportion of rejected hypotheses that are mistakenly rejected 
%            (i.e., the null hypothesis is actually true for those tests). 
%            FDR is a somewhat less conservative/more powerful method for 
%            correcting for multiple comparisons than procedures like Bonferroni
%            correction that provide strong control of the family-wise
%            error rate (i.e., the probability that one or more null
%            hypotheses are mistakenly rejected).
%
%            This function also returns the false coverage-statement rate 
%            (FCR)-adjusted selected confidence interval coverage (i.e.,
%            the coverage needed to construct multiple comparison corrected
%            confidence intervals that correspond to the FDR-adjusted p-values).
%
%
% Usage:
%  >> [h, crit_p, adj_ci_cvrg, adj_p]=fdr_bh(pvals,q,method,report);
%
% Required Input:
%   pvals - A vector or matrix (two dimensions or more) containing the
%           p-value of each individual test in a family of tests.
%
% Optional Inputs:
%   q       - The desired false discovery rate. {default: 0.05}
%   method  - ['pdep' or 'dep'] If 'pdep,' the original Bejnamini & Hochberg
%             FDR procedure is used, which is guaranteed to be accurate if
%             the individual tests are independent or positively dependent
%             (e.g., Gaussian variables that are positively correlated or
%             independent).  If 'dep,' the FDR procedure
%             described in Benjamini & Yekutieli (2001) that is guaranteed
%             to be accurate for any test dependency structure (e.g.,
%             Gaussian variables with any covariance matrix) is used. 'dep'
%             is always appropriate to use but is less powerful than 'pdep.'
%             {default: 'pdep'}
%   report  - ['yes' or 'no'] If 'yes', a brief summary of FDR results are
%             output to the MATLAB command line {default: 'no'}
%
%
% Outputs:
%   h       - A binary vector or matrix of the same size as the input "pvals."
%             If the ith element of h is 1, then the test that produced the 
%             ith p-value in pvals is significant (i.e., the null hypothesis
%             of the test is rejected).
%   crit_p  - All uncorrected p-values less than or equal to crit_p are 
%             significant (i.e., their null hypotheses are rejected).  If 
%             no p-values are significant, crit_p=0.
%   adj_ci_cvrg - The FCR-adjusted BH- or BY-selected 
%             confidence interval coverage. For any p-values that 
%             are significant after FDR adjustment, this gives you the
%             proportion of coverage (e.g., 0.99) you should use when generating
%             confidence intervals for those parameters. In other words,
%             this allows you to correct your confidence intervals for
%             multiple comparisons. You can NOT obtain confidence intervals 
%             for non-significant p-values. The adjusted confidence intervals
%             guarantee that the expected FCR is less than or equal to q
%             if using the appropriate FDR control algorithm for the  
%             dependency structure of your data (Benjamini & Yekutieli, 2005).
%             FCR (i.e., false coverage-statement rate) is the proportion 
%             of confidence intervals you construct
%             that miss the true value of the parameter. adj_ci=NaN if no
%             p-values are significant after adjustment.
%   adj_p   - All adjusted p-values less than or equal to q are significant
%             (i.e., their null hypotheses are rejected). Note, adjusted 
%             p-values can be greater than 1.
%
%
% References:
%   Benjamini, Y. & Hochberg, Y. (1995) Controlling the false discovery
%     rate: A practical and powerful approach to multiple testing. Journal
%     of the Royal Statistical Society, Series B (Methodological). 57(1),
%     289-300.
%
%   Benjamini, Y. & Yekutieli, D. (2001) The control of the false discovery
%     rate in multiple testing under dependency. The Annals of Statistics.
%     29(4), 1165-1188.
%
%   Benjamini, Y., & Yekutieli, D. (2005). False discovery rate?adjusted 
%     multiple confidence intervals for selected parameters. Journal of the 
%     American Statistical Association, 100(469), 71?81. doi:10.1198/016214504000001907
%
%
% Example:
%  nullVars=randn(12,15);
%  [~, p_null]=ttest(nullVars); %15 tests where the null hypothesis
%  %is true
%  effectVars=randn(12,5)+1;
%  [~, p_effect]=ttest(effectVars); %5 tests where the null
%  %hypothesis is false
%  [h, crit_p, adj_ci_cvrg, adj_p]=fdr_bh([p_null p_effect],.05,'pdep','yes');
%  data=[nullVars effectVars];
%  fcr_adj_cis=NaN*zeros(2,20); %initialize confidence interval bounds to NaN
%  if ~isnan(adj_ci_cvrg),
%     sigIds=find(h);
%     fcr_adj_cis(:,sigIds)=tCIs(data(:,sigIds),adj_ci_cvrg); % tCIs.m is available on the
%     %Mathworks File Exchagne
%  end
%
%
% For a review of false discovery rate control and other contemporary
% techniques for correcting for multiple comparisons see:
%
%   Groppe, D.M., Urbach, T.P., & Kutas, M. (2011) Mass univariate analysis 
% of event-related brain potentials/fields I: A critical tutorial review. 
% Psychophysiology, 48(12) pp. 1711-1725, DOI: 10.1111/j.1469-8986.2011.01273.x 
% http://www.cogsci.ucsd.edu/~dgroppe/PUBLICATIONS/mass_uni_preprint1.pdf
%
%
% For a review of FCR-adjusted confidence intervals (CIs) and other techniques 
% for adjusting CIs for multiple comparisons see:
%
%   Groppe, D.M. (in press) Combating the scientific decline effect with 
% confidence (intervals). Psychophysiology.
% http://biorxiv.org/content/biorxiv/early/2015/12/10/034074.full.pdf
%
%
% Author:
% David M. Groppe
% Kutaslab
% Dept. of Cognitive Science
% University of California, San Diego
% March 24, 2010

%%%%%%%%%%%%%%%% REVISION LOG %%%%%%%%%%%%%%%%%
%
% 5/7/2010-Added FDR adjusted p-values
% 5/14/2013- D.H.J. Poot, Erasmus MC, improved run-time complexity
% 10/2015- Now returns FCR adjusted confidence intervals

function [h, crit_p, adj_ci_cvrg, adj_p]=fdr_bh(pvals,q,method,report)

if nargin<1,
    error('You need to provide a vector or matrix of p-values.');
else
    if ~isempty(find(pvals<0,1)),
        error('Some p-values are less than 0.');
    elseif ~isempty(find(pvals>1,1)),
        error('Some p-values are greater than 1.');
    end
end

if nargin<2,
    q=.05;
end

if nargin<3,
    method='pdep';
end

if nargin<4,
    report='no';
end

s=size(pvals);
if (length(s)>2) || s(1)>1,
    [p_sorted, sort_ids]=sort(reshape(pvals,1,prod(s)));
else
    %p-values are already a row vector
    [p_sorted, sort_ids]=sort(pvals);
end
[dummy, unsort_ids]=sort(sort_ids); %indexes to return p_sorted to pvals order
m=length(p_sorted); %number of tests

if strcmpi(method,'pdep'),
    %BH procedure for independence or positive dependence
    thresh=(1:m)*q/m;
    wtd_p=m*p_sorted./(1:m);
    
elseif strcmpi(method,'dep')
    %BH procedure for any dependency structure
    denom=m*sum(1./(1:m));
    thresh=(1:m)*q/denom;
    wtd_p=denom*p_sorted./[1:m];
    %Note, it can produce adjusted p-values greater than 1!
    %compute adjusted p-values
else
    error('Argument ''method'' needs to be ''pdep'' or ''dep''.');
end

if nargout>3,
    %compute adjusted p-values; This can be a bit computationally intensive
    adj_p=zeros(1,m)*NaN;
    [wtd_p_sorted, wtd_p_sindex] = sort( wtd_p );
    nextfill = 1;
    for k = 1 : m
        if wtd_p_sindex(k)>=nextfill
            adj_p(nextfill:wtd_p_sindex(k)) = wtd_p_sorted(k);
            nextfill = wtd_p_sindex(k)+1;
            if nextfill>m
                break;
            end;
        end;
    end;
    adj_p=reshape(adj_p(unsort_ids),s);
end

rej=p_sorted<=thresh;
max_id=find(rej,1,'last'); %find greatest significant pvalue
if isempty(max_id),
    crit_p=0;
    h=pvals*0;
    adj_ci_cvrg=NaN;
else
    crit_p=p_sorted(max_id);
    h=pvals<=crit_p;
    adj_ci_cvrg=1-thresh(max_id);
end

if strcmpi(report,'yes'),
    n_sig=sum(p_sorted<=crit_p);
    if n_sig==1,
        fprintf('Out of %d tests, %d is significant using a false discovery rate of %f.\n',m,n_sig,q);
    else
        fprintf('Out of %d tests, %d are significant using a false discovery rate of %f.\n',m,n_sig,q);
    end
    if strcmpi(method,'pdep'),
        fprintf('FDR/FCR procedure used is guaranteed valid for independent or positively dependent tests.\n');
    else
        fprintf('FDR/FCR procedure used is guaranteed valid for independent or dependent tests.\n');
    end
end
end