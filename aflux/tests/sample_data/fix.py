with open('sample_size_10.dat') as f:
    for l in f:
        spl = l.strip().split()
        print '%s\t%s\t%s\t%s\t' % (spl[0], spl[1], spl[2], spl[3])
