function BASE_PATH = functionSemantic_similaity_env_setup(SYSTEM_PLATFORM, addPath)
%% START >> Select platform
if SYSTEM_PLATFORM == 1
    %For linux laptop
    BASE_PATH = '/home/omy/Documents/omkar-server-backup-20feb2017/Documents/study/phd-research/';
    if addPath
        addpath(genpath(sprintf('%s/codes/third-party-softwares/ocas/libocas_v097/', BASE_PATH)));disp('Enabling ocas/libocas_v097');        
        addpath(genpath(sprintf('%s/codes/third-party-softwares-codes/svm/liblinear', BASE_PATH)));disp('Enabling ocas/liblinear');        
        run(sprintf('%s/codes/third-party-softwares-codes/VL-feat/VLFEATROOT/vlfeat-0.9.20/toolbox/vl_setup', BASE_PATH));disp('Enabling VL feat toolbox');
        addpath(genpath(sprintf('%s/codes/third-party-softwares-codes/t-sne/', BASE_PATH)));disp('Enabling t-SNE');        
    end
elseif SYSTEM_PLATFORM == 3
    %For linux desktop
    BASE_PATH = '/nfs4/omkar/Documents/study/phd-research/';
    if addPath
        addpath(genpath(sprintf('%s/codes/third-party-softwares/ocas/libocas_v097/', BASE_PATH)));disp('Enabling ocas/libocas_v097');        
        addpath(genpath(sprintf('%s/codes/third-party-softwares-codes/svm/liblinear', BASE_PATH)));disp('Enabling ocas/liblinear');        
        run(sprintf('%s/codes/third-party-softwares-codes/VL-feat/VLFEATROOT/vlfeat-0.9.20/toolbox/vl_setup', BASE_PATH));disp('Enabling VL feat toolbox');
        addpath(genpath(sprintf('%s/codes/third-party-softwares-codes/t-sne/', BASE_PATH)));disp('Enabling t-SNE');        
    end
elseif SYSTEM_PLATFORM == 2
    
else
    % For windows desktop
    BASE_PATH = 'E:\omkar-work\study\phd-research\'
    if addPath
        %addpath(genpath(sprintf('%s/codes/third-party-softwares/ocas/libocas_v097/', BASE_PATH)));disp('Enabling ocas/libocas_v097');        
        addpath(genpath(sprintf('%s/codes/third-party-softwares-codes\svm\libSVM\libsvm-3.21\windows\', BASE_PATH)));disp('Enabling ocas/liblinear');        
        run(sprintf('%s/codes/third-party-softwares-codes/VL-feat/VLFEATROOT/vlfeat-0.9.20/toolbox/vl_setup', BASE_PATH));disp('Enabling VL feat toolbox');
        addpath(genpath(sprintf('%s/codes/third-party-softwares-codes/t-sne/', BASE_PATH)));disp('Enabling t-SNE');        
    end
end
%% END >> Select platform