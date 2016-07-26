eighths = [' ','\u258F','\u258E','\u258D','\u258C','\u258B','\u258A','\u2589','\u2588']

def print_progress(prog, total=1.0, length=70, reset=True):
  if reset:
    endChar = '\r'
  else:
    endChar = ''
  if prog > total:
    prog = total
  full_boxes = int((prog / total) * length)
  string = eighths[8]*full_boxes
  if full_boxes != length:
    string += eighths[int(((prog/total)-(full_boxes/length))*8*length)]
    string += ' ' * (length - full_boxes - 1)
  print('|' + string + '| %.1f%%' % (1e2*prog/total), end=endChar) 
