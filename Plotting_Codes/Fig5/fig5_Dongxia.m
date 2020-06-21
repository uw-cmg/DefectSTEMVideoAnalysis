% click 3 excel files to import data into MATLAB
%  Must choose import as Numeric Matrix Instead of Tables
% then run this script to run the files

figure
labelFontSize=35;
ticFontSize=25;

subplot(1,2,1)
boxplot(manualboxplotdata,{'1.0','1.5','2.0','2.5'})
ylim([0 25])
xlabel('Dose (dpa)', 'FontSize', labelFontSize), ylabel('Median Size (nm)', 'FontSize', labelFontSize)
%%%%%%%%%%%%% Changing Tic Label %%%%%%%%%%%%%%%%%%%%
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',ticFontSize,'FontWeight','bold');
a = get(gca,'YTickLabel'); 
set(gca,'YTickLabel',a,'fontsize',ticFontSize,'FontWeight','bold');
title('Ground Truth Labeling Data','FontSize', 22)
set(findobj(gca,'type','line'),'linew',3)



subplot(1,2,2)
boxplot(watershedboxplotdata,{'1.0','1.5','2.0','2.5'})
ylim([0 25])
xlabel('Dose (dpa)', 'FontSize', labelFontSize), ylabel('Median Size (nm)', 'FontSize', labelFontSize)
%%%%%%%%%%%%% Changing Tic Label %%%%%%%%%%%%%%%%%%%%
a = get(gca,'XTickLabel');  
set(gca,'XTickLabel',a,'fontsize',ticFontSize,'FontWeight','bold');
a = get(gca,'YTickLabel'); 
set(gca,'YTickLabel',a,'fontsize',ticFontSize,'FontWeight','bold');
title('Machine Learning Analysis Data','FontSize', 22)
set(findobj(gca,'type','line'),'linew',3)

set(gcf, 'Position', [0 0 1200 600])


