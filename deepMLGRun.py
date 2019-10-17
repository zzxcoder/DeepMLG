# -*- coding: utf-8 -*-
import sys
import os
import os.path
import argparse

from nurall.lstm.lstmnet import neull.klst


if __name__ == "main":
    ## 终端命令行参数
    parser = argparse.ArgumentParser(description = '');
    parser.add_argument();
    
    
    
    
    args = parser.parse_args();
    
    ## 判断输入的命令参数是否正确
    if len(sys.argv) == 1:
        print("For help on the usage of this program, please use the option -h."); exit(1)
    
    
    
    ## 根据命令行的任务类型创建session（train 或者 test）
    if args.train:
        train_cfg_param = getConfigParameter(args.train_cfg_file);
        session = TrainSession(train_cfg_param);
    elif args.test:
        test_cfg_param = getConfigParameter(args.test_cfg_file);
        session = TestSession(test_cfg_param);
        
    ## 创建输出目录和日志器
    session.create_output_folder();
    session.create_logger();
    log = session.get_logger();
    
    try:
        ## 执行session
        session.run();
        
    except (Exception, KeyboardInterrupt) as e:
        ## 输出异常信息
        log.print();
        log.print("ERROR: Caught exception from main process: " + str(e) );
        
    
    log.print("Finished.");
    
    
