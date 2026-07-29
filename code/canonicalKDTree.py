from pyspark.sql.dataframe import DataFrame
def canonicalKDTree(self, variables, depth = 10, accuracy = 100):
    
    class KDNode:        
        def __init__(self, median, variables, axis, level, leaf, parent=None, left=None, right=None):
            self.median = median
            self.axis = variables[axis]
            self.level = level
            self.leaf = leaf
            self.parent = parent
            self.left = left
            self.right = right
            
    def cycle(iterable):
        saved = []
        for element in iterable:
            yield element
            saved.append(element)
        while saved:
            for element in saved:
                yield element

    def islice(iterable, stop):
        count = 0
        for element in iterable:
            if count >= stop:
                return
            yield element
            count += 1
        
    def updateleaf(row, var, medians):
        from pyspark.sql import Row
        r = row.asDict()
        r['leaf'] *= 2
        if row[var] > medians[row['leaf']]:
            r['leaf'] += 1
        return Row(**r)

    def growBranch(data, depth, var, accuracy):
        from pyspark.sql.functions import lit, percentile_approx
        medians = data.groupBy("leaf").agg(percentile_approx(var, 0.5, lit(accuracy)).alias("median")).sort("leaf").rdd.map(lambda row: row["median"]).collect()
        data = data.rdd.map(lambda row: updateleaf(row, var, medians)).toDF()
        return medians, data
    
    def f_assign_leaf(row, tree, height):
        from pyspark.sql import Row
        i = row['leaf']
        for d in range(height, len(tree)):
            if row[tree[d]['Splitting variable']] <= tree[d]['Splitting points'][i]:
                i = 2*i
            else:
                i = 2*i+1
        rowDict = row.asDict()
        rowDict['leaf'] = i
        return(Row(**rowDict))
    
    def bfs2dfs(index, depth, variables, flat_medians, parent=None):
        if index >= len(flat_medians) or flat_medians[index] is None:
            return None
        axis = depth % len(variables)
        leaf = index - (2 ** depth - 1)
        node = KDNode(
            median=flat_medians[index],
            variables = variables,
            axis=axis,
            level=depth,
            leaf=leaf,
            parent=parent
        )
        node.left = bfs2dfs(2 * index + 1, depth + 1, variables, flat_medians, parent=node)
        node.right = bfs2dfs(2 * index + 2, depth + 1, variables, flat_medians, parent=node)
        return node
        
    import numpy, pandas, pyspark, time
    from pyspark.sql.functions import lit, col, percentile, percentile_approx
    bfs = []
    data = self.withColumn('leaf', lit(0))
    for depth, variable_axis in list(enumerate(islice(cycle(variables), depth))):
        medians, data = growBranch(data, depth, variable_axis, accuracy)
        bfs = bfs + [{'Depth': depth, 'Splitting variable': variable_axis, 'Splitting points': numpy.array(medians)}]
    flat_medians = [float(m) for level in bfs for m in level['Splitting points']]
    dfs = bfs2dfs(0, 0, variables, flat_medians)
    tree = {'dfs': dfs, 'bfs': bfs}
    return tree

setattr(DataFrame, "canonicalKDTree", canonicalKDTree)

def treeLeafCounts(self, tree):
    def f_assign_leaf(row, root):
        from pyspark.sql import Row
        node = root
        while node.left is not None:
            axis = node.axis
            leaf = node.leaf
            if row[axis] <= node.median:
                node = node.left
            else:
                node = node.right
        axis = node.axis
        leaf = 2*node.leaf
        if row[axis] > node.median:
            leaf += 1
        leafDict = {'leaf': leaf}
        return(Row(**leafDict))
    root = tree['dfs']
    dataLeafs = self.rdd.map(lambda row: f_assign_leaf(row, root)).toDF().groupby('leaf').count().sort('leaf').toPandas()
    return dataLeafs

setattr(DataFrame, "treeLeafCounts", treeLeafCounts)

def treePrecision(self, tree):
    import numpy
    def f_assign_leaf(row, root):
        from pyspark.sql import Row
        node = root
        while node.left is not None:
            axis = node.axis
            leaf = node.leaf
            if row[axis] <= node.median:
                node = node.left
            else:
                node = node.right
        axis = node.axis
        leaf = 2*node.leaf
        if row[axis] > node.median:
            leaf += 1
        leafDict = {'leaf': leaf}
        return(Row(**leafDict))
    root = tree['dfs']
    dataLeafs = self.rdd.map(lambda row: f_assign_leaf(row, root)).toDF().groupby('leaf').count().sort('leaf').toPandas()
    C = dataLeafs['count'].values
    precision = - numpy.log(float(sum(abs(C - sum(C)/len(C)))/sum(C)))
    return precision

setattr(DataFrame, "treePrecision", treePrecision)