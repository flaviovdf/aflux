with open('/var/tmp/example.dat') as ex:
    for l in ex:
        h, s, d, c = l.split('\t')
        c = int(c)
        if s != d and c >= 50:
            print l,
