from pyspark.sql.dataframe import DataFrame

def pKDTree(self, variables, depth, lamda = 4, sigma = 32):
    from pyspark.sql.functions import lit
    
    class KDNode:        
        def __init__(self, median, variables, axis, level, leaf, parent=None, left=None, right=None):
            self.median = median
            self.axis = variables[axis]
            self.level = level
            self.leaf = leaf
            self.parent = parent
            self.left = left
            self.right = right
            
    def f_branch_merge_local(row):
        rowDict = row.asDict()
        key = rowDict.pop('leaf')
        value = [rowDict]
        return (key, value)

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

    def f_branch_merge(x):
        import numpy
        return((x[0], numpy.array([dict(x[1])[i] for i in range(len(x[1]))]).ravel()))

    def assignLeaf(data, tree, height):
        data = data.rdd.map(lambda row: f_assign_leaf(row, tree, height)).toDF()
        return data

    def getTree(data, localDepth, var_index, variables):
        import numpy
        import pandas
        leaf = data.loc[0, 'leaf']
        tree = [[]] * localDepth
        for d in range(localDepth):
            greater = 0
            tree[d] = []
            for k in range(leaf*2**d, (leaf+1)*2**d):
                median = numpy.median(data.loc[data['leaf'] == k, variables[var_index]].values)
                tree[d].append(median)
                greater += ((data['leaf'] == k) & (data[variables[var_index]] > median)).values
            var_index = (var_index + 1) % len(variables)
            data.leaf = 2 * data.leaf + greater
        return tree

    def addLocalBranches(data, tree, localDepth, variables):
        tree_depth = len(tree)
        start_var_index = tree_depth % len(variables)
        branches_list = data.rdd.map(lambda x: f_branch_merge_local(x)).reduceByKey(lambda x, y: x+y).flatMap(lambda x: localBranch(x, tree_depth, localDepth, start_var_index, variables)).reduceByKey(lambda x, y: x+y).map(lambda x: f_branch_merge(x)).collect()
        branches_dict = {r[0][0]: {'Depth' : r[0][0], 'Splitting variable': r[0][1], 'Splitting points': r[1]} for r in branches_list}
        for d in range(len(branches_dict)):
            tree.append(branches_dict[d+tree_depth])
        return(tree)

    def localBranch(x, tree_depth, localDepth, start_var_index, variables):
        import numpy
        import pandas
        p = len(variables)
        data = pandas.DataFrame(x[1]).assign(leaf = x[0])
        subtree = getTree(data, localDepth, start_var_index, variables)
        branch = [((tree_depth + d, variables[(start_var_index + d) % p]), [(x[0], numpy.array(subtree[d]))]) for d in range(localDepth)]
        return(branch)

    def getSample(data, lamda, localDepth, sigma):
        from pyspark.sql.functions import rand, row_number, col
        from pyspark.sql.window import Window
        lamda = min(lamda, localDepth)
        sampleSize = (2**lamda)*sigma
        buffer = 2*sigma
        leafCounts = data.groupBy("leaf").count().collect()
        leafFractions = {row['leaf']: min(1.0, (sampleSize + buffer) / row['count']) for row in leafCounts}
        sampleData = data.sampleBy("leaf", leafFractions, seed=42).withColumn("rand", rand())
        window = Window.partitionBy("leaf").orderBy("rand")
        sampleData = sampleData.withColumn("rank", row_number().over(window)).filter(col("rank") <= sampleSize).drop("rand", "rank")
        return sampleData
    
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
        
    data = self.select(*variables).withColumn("leaf", lit(0))
    bfs = []
    height = 0
    localDepthList = [lamda] * int(depth/lamda) + [depth%lamda]
    for localDepth in localDepthList:
        sampleData = getSample(data, lamda, localDepth, sigma)
        bfs = addLocalBranches(sampleData, bfs, localDepth, variables)
        data = assignLeaf(data, bfs, height)
        height += localDepth
    flat_medians = [float(m) for level in bfs for m in level['Splitting points']]
    dfs = bfs2dfs(0, 0, variables, flat_medians)
    tree = {'dfs': dfs, 'bfs': bfs}
    return tree

setattr(DataFrame, "pKDTree", pKDTree)

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