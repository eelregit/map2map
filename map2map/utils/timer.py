import numpy as np
import time
import sys

class Timer(object) :

    def __init__(self, t_print_length = 0) :
        self.start_time = time.time()
        self.last_time = self.start_time
        self.print_length = t_print_length

    # function for formatting timing strings, hh/mm/ss, where hh and mm are %02d ints
    # for hours and minutes and ss is a %.2f float for seconds
    def formatTime(self, t_time) :
        h = int(np.floor(t_time/60./60.))
        m = int(np.floor((t_time - h*60.*60.)/60.))
        s = t_time - m*60. - h*60*60.
        return "%02d:%02d:%05.2f" % (h, m, s)

    def getTimes(self) :
        this_time = time.time()
        task_time = this_time - self.last_time
        full_time = this_time - self.start_time
        self.last_time = this_time
        return "[" + self.formatTime(task_time) + ", " + self.formatTime(full_time) + "]"

    def getTotalTime(self) :
        return "[" + self.formatTime(time.time() - self.start_time) + "]"

    def printStart(self, t_string, t_end = '') :
        self.current_length = len(t_string)
        print(t_string, end = t_end, flush = True)
        sys.stdout.flush()
        return

    def printDone(self, t_num_spaces = 0) :
        if t_num_spaces != 0 :
            print(" " * num_spaces + "done " + self.getTimes(), flush = True)
        elif self.print_length - self.current_length > 0 :
            print(" " * (self.print_length - self.current_length) + "done " + self.getTimes(), flush = True)
        else :
            print("done " + self.getTimes(), flush = True)
        sys.stdout.flush()
        return
