import heapq


def perform_select(fits, n_chrom):
    selected_chroms = list(map(fits.index, heapq.nlargest(n_chrom, fits)))
    return selected_chroms
