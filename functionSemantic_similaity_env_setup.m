function BASE_PATH = functionSemantic_similaity_env_setup(SYSTEM_PLATFORM, addPath)

%% START >> Select platform
if SYSTEM_PLATFORM == 1
    %For laptop
    BASE_PATH = '/home/omy/Documents/omkar-server-backup-20feb2017/Documents/study/phd-research/';
    if addPath
        addpath(genpath(sprintf('%s/codes/third-party-softwares/ocas/libocas_v097/', BASE_PATH)));
        addpath(genpath(sprintf('%s/codes/third-party-softwares-codes/svm/liblinear', BASE_PATH)));
        run(sprintf('%s/codes/third-party-softwares-codes/VL-feat/VLFEATROOT/vlfeat-0.9.20/toolbox/vl_setup', BASE_PATH));
        disp('Enabling VL feat toolbox');
        addpath(genpath(sprintf('%s/codes/third-party-softwares-codes/t-sne/', BASE_PATH)));
    end
elseif SYSTEM_PLATFORM == 3
    %For desktop
    BASE_PATH = '/nfs4/omkar/Documents/study/phd-research/';
    if addPath
        addpath(genpath(sprintf('%s/codes/third-party-softwares/ocas/libocas_v097/', BASE_PATH)));
        addpath(genpath(sprintf('%s/codes/third-party-softwares-codes/svm/liblinear', BASE_PATH)));
        run(sprintf('%s/codes/third-party-softwares-codes/VL-feat/VLFEATROOT/vlfeat-0.9.20/toolbox/vl_setup', BASE_PATH));
        disp('Enabling VL feat toolbox');
        addpath(genpath(sprintf('%s/codes/third-party-softwares-codes/t-sne/', BASE_PATH)));
    end
elseif SYSTEM_PLATFORM == 2
    
else
    BASE_PATH = 'E:\omkar-work\study\';
    if addPath
        addpath(genpath(sprintf('%s\codes\third-party-softwares-codes\svm\liblinear\liblinear-2.11\windows\', BASE_PATH)));
        run(sprintf('%s/codes/third-party-softwares-codes/VL-feat/VLFEATROOT/vlfeat-0.9.20/toolbox/vl_setup', BASE_PATH));
        disp('Enabling VL feat toolbox');
        addpath(genpath(sprintf('%s/codes/third-party-softwares-codes/t-sne/', BASE_PATH)));
    end
end
%% END >> Select platform


