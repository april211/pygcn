import time
import logging
from logging import handlers


time_string = time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime())

filename = "./logs/log-{}.csv".format(time_string)

class MyLogger(object):

    str_level_map = {
        'debug'     : logging.DEBUG,
        'info'      : logging.INFO,
        'warning'   : logging.WARNING,
        'error'     : logging.ERROR,
        'critical'  : logging.CRITICAL
    }

    def __init__(self,
                 filename=filename,
                 level='debug',
                 when='D',
                 backupCount=0,
                 fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        
        self.logger = logging.getLogger(filename)
        self.logger.setLevel(self.str_level_map.get(level))

        # set format for screen output
        format_str = logging.Formatter(fmt)                             
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        
        # write logs in files
        th = handlers.TimedRotatingFileHandler(filename=filename,
                                               when=when,
                                               backupCount=backupCount,
                                               encoding='utf-8')
        
        # set format for log files
        th.setFormatter(format_str)

        # direct logger output to screen & log files
        self.logger.addHandler(sh)
        self.logger.addHandler(th)

mylogger = MyLogger(level="debug")

