from pyspark.sql.dataframe import DataFrame

def epstatKDTree(self, variables, J = [2, 4, 4], depth = None, batch_size = None, local_depth = None):
    
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

    def batched(iterable, n):
        batch = []
        for element in iterable:
            batch.append(element)
            if len(batch) == n:
                yield tuple(batch)
                batch = []
        if batch:
            yield tuple(batch) 
        
    def ChevTrA(T):
        J = T.shape[0]
        E = T.copy()
        O = T.copy()
        for j in range(J):
            T[j] = O[0]
            for k in range(J-j-1):
                E[k] = 2 * O[k+1] - E[k]
                O[k] = 2 * E[k] - O[k]
        return(T)

    def ChevTrB(T):
        J = T.shape[0]
        E = T.copy()
        O = T[1:].copy()
        for j in range(1, J):
            E[j] = T[j]
            O[j-1] = T[j]
        for j in range(int(J/2)):
            T[2*j] = E[0]
            T[2*j+1] = O[0]
            for k in range(J-2*j-2):
                E[k] = 2 * O[k+1] - E[k]
            for k in range(J-2*j-3):
                O[k] = 2 * E[k+1] - O[k]
        if J % 2 == 1:
            T[J-1] = E[0]
        return(T)

    def SineTr(T):
        J = T.shape[0]
        S = T.copy()
        for k in range(1, J):
            for j in range(k, J):
                S[j] = T[j-1] - T[j]
            for j in range(k, J):
                T[j] = S[j]
        return(T)

    def OddNeg(T):
        J = T.shape[0]
        for j in range(J):
            T[j] = (-1)**j * T[j]
        return(T)

    def CosProd2Cos(C):
        from numpy import array
        if len(C.shape) == 2:
            C_res = [None] * C.shape[-1]
            C_add = C[:, 0]
            C_res[0] = C_add
            for j in range(1, C.shape[-1]):
                C_add = 2 * C[:, j] - C_add[::-1]
                C_res[j] = C_add
            return(array(C_res).ravel())
        else:
            C_res = [None] * C.shape[-1]
            C_add = CosProd2Cos(C[..., 0])
            C_res[0] = C_add
            for j in range(1, C.shape[-1]):
                C_add = 2 * CosProd2Cos(C[..., j]) - C_add[::-1]
                C_res[j] = C_add
            return(array(C_res).ravel())

    def SinProd2Sin(C):
        from numpy import array
        if len(C.shape) == 2:
            C_res = [None] * C.shape[-1]
            C_add = C[:, 0]
            C_res[0] = C_add
            for j in range(1, C.shape[-1]):
                C_add = 2 * C[:, j] + C_add[::-1]
                C_res[j] = C_add
            return(array(C_res).ravel())
        else:
            C_res = [None] * C.shape[-1]
            C_add = SinProd2Sin(C[..., 0])
            C_res[0] = C_add
            for j in range(1, C.shape[-1]):
                C_add = 2 * SinProd2Sin(C[..., j]) + C_add[::-1]
                C_res[j] = C_add
            return(array(C_res).ravel())

    def TrigProd2Trig(C, J):
        from numpy import delete, apply_along_axis, insert, stack
        J_ = J.copy()
        J_[0] = 2 * J[0]
        C_0 = C[0]
        C = delete(C, 0).reshape(J_)
        C_cos = C[range(0, J_[0], 2), ...]
        C_cos = apply_along_axis(ChevTrA, 0, C_cos)
        for d in range(1, len(J)):
            C_cos = apply_along_axis(ChevTrB, d, C_cos)
        C_cos = CosProd2Cos(C_cos)
        C_sin = C[range(1, J_[0], 2), ...]
        C_sin = apply_along_axis(SineTr, 0, C_sin)
        C_sin = apply_along_axis(ChevTrA, 0, C_sin)
        C_sin = apply_along_axis(OddNeg, 0, C_sin)
        for d in range(1, len(J)):
            C_sin = apply_along_axis(ChevTrB, d, C_sin)
        C_sin = SinProd2Sin(C_sin)
        return(insert(stack((C_cos, C_sin), axis = 1).ravel(), 0, C_0))

    def wepTransform(C, J):
        from numpy import log, prod, repeat, apply_along_axis
        P = int(log(C.shape[0]) / log(2 * prod(J) + 1))
        C = C.reshape(tuple(repeat(2 * prod(J) + 1, P)))
        def f(C):
            return(TrigProd2Trig(C, J))
        for d in range(P):
            C = apply_along_axis(f, d, C)
        return(C)

    def getMultiplierKDtree(a, b, J):
        from numpy import zeros, pi, sin, cos
        F = zeros(2*J+1)
        a_inc = 2*a
        b_inc = 2*b
        F[0] = 1
        if a > 0:
            F[0] -= .5
            for j in range(1, 2 * J, 2):
                F[j] -= 2 / (pi * j) * sin(a)
                F[j+1] += 2 / (pi * j) * cos(a)
                a += a_inc
        if b < 1:
            F[0] -= .5
            for j in range(1, 2 * J, 2):
                F[j] += 2 / (pi * j) * sin(b)
                F[j+1] -= 2 / (pi * j) * cos(b)
                b += b_inc
        return(F)

    def wep2multiplier(wep, lower, upper, splitting_axis):
        from numpy import outer, rollaxis
        P = len(wep.shape)
        J = int((wep.shape[0] - 1) / 2)
        multiplier_array = 1
        for p in range(P):
            if p != splitting_axis:
                multiplier_array = outer(multiplier_array, getMultiplierKDtree(lower[p], upper[p], J))
        multiplier_array = multiplier_array.ravel()
        multiplier = rollaxis(wep, splitting_axis).reshape(2*J+1, (2*J+1) ** (P - 1)).dot(multiplier_array)
        return(multiplier)

    def wep2median(wep, lower, upper, splitting_axis):
        from scipy.optimize import brentq
        P = len(wep.shape)
        J = int((wep.shape[0] - 1) / 2)
        if P == 1:
            multiplier = wep.ravel()
        else:
            multiplier = wep2multiplier(wep, lower, upper, splitting_axis)
        def f_median(m):
            return(multiplier.dot(getMultiplierKDtree(lower[splitting_axis], m, J) - getMultiplierKDtree(m, upper[splitting_axis], J)))
        median = brentq(lambda m: f_median(m), lower[splitting_axis], upper[splitting_axis])
        return(median)

    def wep2unscaledSubNeighborhoodsInfo(wep, variables_unique, unscaled_neighborhoods, splitting_axis, boundaries):
        from numpy import array, empty
        P = len(wep.shape)
        J = int((wep.shape[0] - 1) / 2)
        I = unscaled_neighborhoods.shape[0]
        lower = array([boundaries['lower'][variable] for variable in variables_unique])
        upper = array([boundaries['upper'][variable] for variable in variables_unique])
        new_unscaled_neighborhoods = empty((2*I, 2, P))
        new_medians = empty(I)
        for i in range(I):
            median = wep2median(wep, unscaled_neighborhoods[i][0], unscaled_neighborhoods[i][1], splitting_axis)
            new_medians[i] = median * (upper[splitting_axis] - lower[splitting_axis]) + lower[splitting_axis]
            new_unscaled_neighborhoods[2*i][0] = unscaled_neighborhoods[i][0]
            new_unscaled_neighborhoods[2*i][1] = unscaled_neighborhoods[i][1]
            new_unscaled_neighborhoods[2*i+1][0] = unscaled_neighborhoods[i][0]
            new_unscaled_neighborhoods[2*i+1][1] = unscaled_neighborhoods[i][1]
            new_unscaled_neighborhoods[2*i][1][splitting_axis] = median
            new_unscaled_neighborhoods[2*i+1][0][splitting_axis] = median
        unscaled_neighborhoods_info = {'unscaled_neighborhoods' : new_unscaled_neighborhoods, 'medians' : new_medians}
        return(unscaled_neighborhoods_info)

    def wep2tree(wep, variables_unique, indices_tuple, boundaries):
        from numpy import zeros, ones
        P = len(wep.shape)
        D = len(indices_tuple)
        unscaled_neighborhoods = zeros((1, 2, P))
        unscaled_neighborhoods[0][1] = ones(P)
        medians = [None] * D
        splitting_variables = [None] * D
        for d in range(D):
            splitting_axis = indices_tuple[d]
            unscaled_neighborhoods_info = wep2unscaledSubNeighborhoodsInfo(wep, variables_unique, unscaled_neighborhoods, splitting_axis, boundaries)
            unscaled_neighborhoods = unscaled_neighborhoods_info['unscaled_neighborhoods']
            medians[d] = unscaled_neighborhoods_info['medians']
            splitting_variables[d] = variables_unique[splitting_axis]
        tree = [{'Depth' : d, 'Splitting variable': splitting_variables[d], 'Splitting points': medians[d]} for d in range(D)]
        return(tree)

    def wep2branch(wep, variables_unique, indices_tuple, branch_index, tree_depth, branch_boundaries_list):
        from numpy import zeros, ones
        boundaries = {'lower': branch_boundaries_list[branch_index]['lower'].copy(), 'upper': branch_boundaries_list[branch_index]['upper'].copy()}
        P = len(wep.shape)
        D = len(indices_tuple)
        unscaled_neighborhoods = zeros((1, 2, P))
        unscaled_neighborhoods[0][1] = ones(P)
        medians = [None] * D
        splitting_variables = [None] * D
        for d in range(D):
            splitting_axis = indices_tuple[d]
            unscaled_neighborhoods_info = wep2unscaledSubNeighborhoodsInfo(wep, variables_unique, unscaled_neighborhoods, splitting_axis, boundaries)
            unscaled_neighborhoods = unscaled_neighborhoods_info['unscaled_neighborhoods']
            medians[d] = unscaled_neighborhoods_info['medians']
            splitting_variables[d] = variables_unique[splitting_axis]
        branch = [((d + tree_depth, splitting_variables[d]), [(branch_index, medians[d])]) for d in range(D)]
        return(branch)
    
    def getTree(data, local_depth, var_index, variables):
        from numpy import median
        leaf = data.loc[0, 'leaf']
        tree = [[]] * local_depth
        for d in range(local_depth):
            greater = 0
            tree[d] = []
            for k in range(leaf*2**d, (leaf+1)*2**d):
                medianCond = median(data.loc[data['leaf'] == k, variables[var_index]].values)
                tree[d].append(medianCond)
                greater += ((data['leaf'] == k) & (data[variables[var_index]] > medianCond)).values
            var_index = (var_index + 1) % len(variables)
            data.leaf = 2 * data.leaf + greater
        return tree

    def localBranch(x, tree_depth, local_depth, start_var_index, variables):
        from numpy import array
        from pandas import DataFrame
        p = len(variables)
        data = DataFrame(x[1]).assign(leaf = x[0])
        subtree = getTree(data, local_depth, start_var_index, variables)
        branch = [((tree_depth + d, variables[(start_var_index + d) % p]), [(x[0], array(subtree[d]))]) for d in range(local_depth)]
        return(branch)

    def f_tree(row, variables_unique, boundaries, J):
        from numpy import empty, cos, sqrt, sign, outer, insert
        lower, upper = boundaries['lower'], boundaries['upper']
        x = [(row[variable] - lower[variable]) / (upper[variable] - lower[variable]) for variable in variables_unique]
        P = len(x)
        K = len(J)
        C = [None] * K
        C[0] = empty(2 * J[0])
        for k in range(1, K):
            C[k] = empty(J[k])
            C[k][0] = 1
        CC = 1
        for xx in x:
            c = cos(xx)
            m = c*c
            s = sqrt(1 - m) * sign(xx)
            C[0][:2] = c, s
            for j in range(1, J[0]):
                c *= m
                s *= m
                C[0][2*j: 2*j+2] = c, s
            cc = C[0]
            X = 2*xx
            for k in range(1, K):
                X *= J[k-1]
                c = 1
                m = cos(X)
                for j in range(1, J[k]):
                    c *= m
                    C[k][j] = c
                cc = outer(cc, C[k])
            cc = insert(cc.ravel(), 0, 1.)
            CC = outer(CC, cc)
        return(CC.ravel())
    
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

    def f_branch_sep(row, variables_unique, branch_boundaries_list, J):
        from numpy import empty, cos, sqrt, sign, outer, insert
        i = row['leaf']
        lower = branch_boundaries_list[i]['lower'].copy()
        upper = branch_boundaries_list[i]['upper'].copy()
        x = [(row[variable] - lower[variable]) / (upper[variable] - lower[variable]) for variable in variables_unique]
        P = len(x)
        K = len(J)
        C = [None] * K
        C[0] = empty(2 * J[0])
        for k in range(1, K):
            C[k] = empty(J[k])
            C[k][0] = 1
        CC = 1
        for xx in x:
            c = cos(xx)
            m = c*c
            s = sqrt(1 - m) * sign(xx)
            C[0][:2] = c, s
            for j in range(1, J[0]):
                c *= m
                s *= m
                C[0][2*j: 2*j+2] = c, s
            cc = C[0]
            X = 2*xx
            for k in range(1, K):
                X *= J[k-1]
                c = 1
                m = cos(X)
                for j in range(1, J[k]):
                    c *= m
                    C[k][j] = c
                cc = outer(cc, C[k])
            cc = insert(cc.ravel(), 0, 1.)
            CC = outer(CC, cc)
        return((i, CC.ravel()))

    def f_branch_compute(x, tree_depth, variables_unique, indices_tuple, branch_boundaries_list, J):
        from numpy import empty
        branch_index = x[0]
        branch_bin_id = dict(reversed(list(enumerate(bin(branch_index)[2:].zfill(tree_depth)))))
        i = 0
        for d in range(tree_depth):
            if branch_bin_id[d] == '0':
                i = 2*i
            else:
                i = 2*i+1
        sep_raw = x[1]
        wep_raw = empty(len(sep_raw))
        wep_raw[0] = 1.
        for j in range(1, len(sep_raw)):
            wep_raw[j] = sep_raw[j] / sep_raw[0]
        wep = wepTransform(wep_raw, J)
        branch = wep2branch(wep, variables_unique, indices_tuple, branch_index, tree_depth, branch_boundaries_list)

        return(branch)
   
    def f_branch_merge_local(row):
        rowDict = row.asDict()
        key = rowDict.pop('leaf')
        value = [rowDict]
        return (key, value)
    
    def f_branch_merge(x):
        from numpy import array
        return((x[0], array([dict(x[1])[i] for i in range(len(x[1]))]).ravel()))
    
    def getBranchBoundariesList(boundaries, tree):
        tree_depth = len(tree)
        branch_boundaries_list = [None] * (2**tree_depth)
        for branch_index in range(len(branch_boundaries_list)):
            lower = boundaries['lower'].copy()
            upper = boundaries['upper'].copy()
            branch_bin_id = dict(reversed(list(enumerate(bin(branch_index)[2:].zfill(tree_depth)))))
            i = 0
            for d in range(tree_depth):
                if branch_bin_id[d] == '0':
                    upper[tree[d]['Splitting variable']] = tree[d]['Splitting points'][i]
                    i = 2*i
                else:
                    lower[tree[d]['Splitting variable']] = tree[d]['Splitting points'][i]
                    i = 2*i+1
            branch_boundaries_list[branch_index] = {'lower' : lower, 'upper' : upper}
        return(branch_boundaries_list)

    def getIndices(variables_tuple):
        variables_unique = tuple(set(variables_tuple))
        indices_tuple = tuple([{variable: i for i, variable in enumerate(variables_unique)}[variable] for variable in variables_tuple])
        return(variables_unique, indices_tuple)

    def buildTree(data, boundaries, variables_tuple, J):
        from numpy import zeros
        from operator import add
        variables_unique, indices_tuple = getIndices(variables_tuple)
        sep_raw = data.rdd.map(lambda row: f_tree(row, variables_unique, boundaries, J)).reduce(add)
        wep_raw = zeros(len(sep_raw))
        wep_raw[0] = 1.
        for j in range(1, len(sep_raw)):
            wep_raw[j] = sep_raw[j] / sep_raw[0]
        wep = wepTransform(wep_raw, J)
        tree = wep2tree(wep, variables_unique, indices_tuple, boundaries)
        return(tree)

    def assignLeaf(data, tree, height):
        data = data.rdd.map(lambda row: f_assign_leaf(row, tree, height)).toDF()
        return data
        
    def addBranches(data, tree, branch_boundaries_list, variables_tuple, J):
        from operator import add
        variables_unique, indices_tuple = getIndices(variables_tuple)
        tree_depth = len(tree)
        sep_rdd = data.rdd.map(lambda row: f_branch_sep(row, variables_unique, branch_boundaries_list, J)).reduceByKey(add)
        branches_list = sep_rdd.flatMap(lambda x: f_branch_compute(x, tree_depth, variables_unique, indices_tuple, branch_boundaries_list, J)).reduceByKey(lambda x, y: x+y).map(lambda x: f_branch_merge(x)).collect()
        branches_dict = {r[0][0]: {'Depth' : r[0][0], 'Splitting variable': r[0][1], 'Splitting points': r[1]} for r in branches_list}
        for d in range(len(branches_dict)):
            tree.append(branches_dict[d+tree_depth])
        return(tree)
        
    def addLocalBranches(data, tree, local_depth, variables):
        tree_depth = len(tree)
        start_var_index = height % len(variables)
        branches_list = data.rdd.map(lambda x: f_branch_merge_local(x)).reduceByKey(lambda x, y: x+y).flatMap(lambda x: localBranch(x, tree_depth, local_depth, start_var_index, variables)).reduceByKey(lambda x, y: x+y).map(lambda x: f_branch_merge(x)).collect()
        branches_dict = {r[0][0]: {'Depth' : r[0][0], 'Splitting variable': r[0][1], 'Splitting points': r[1]} for r in branches_list}
        for d in range(len(branches_dict)):
            tree.append(branches_dict[d+tree_depth])
        return(tree)
    
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
    
    from pyspark.sql.functions import max, min, lit
    data = self.select(*variables).withColumn("leaf", lit(0))
    bounds = data.agg(*[max(variable).alias(f"max_{variable}") for variable in variables] + [min(variable).alias(f"min_{variable}") for variable in variables]).collect()[0]
    boundaries = {'lower': {variable:bounds.asDict()['min_' + variable] for variable in variables}, 'upper': {variable:bounds.asDict()['max_' + variable] for variable in variables}}  
    if batch_size == None:
        variables_tuples = batched(islice(cycle(variables), depth), depth)
    else:
        variables_tuples = batched(islice(cycle(variables), depth), batch_size)
    variables_tuple = next(variables_tuples)
    bfs = buildTree(data, boundaries, variables_tuple, J)
    height = 0
    if batch_size is not None:
        for variables_tuple in variables_tuples:
            data = assignLeaf(data, bfs, height)
            height = len(bfs)            
            branch_boundaries_list = getBranchBoundariesList(boundaries, bfs)
            bfs = addBranches(data, bfs, branch_boundaries_list, variables_tuple, J)
    if local_depth is not None:
        data = assignLeaf(data, bfs, height)
        height = len(bfs)            
        bfs = addLocalBranches(data, bfs, local_depth, variables)
    flat_medians = [float(m) for level in bfs for m in level['Splitting points']]
    dfs = bfs2dfs(0, 0, variables, flat_medians)
    tree = {'dfs': dfs, 'bfs': bfs}
    return tree

setattr(DataFrame, "epstatKDTree", epstatKDTree)

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

def treeLeafCells(self, tree):
    from pyspark.sql.window import Window
    from pyspark.sql.functions import row_number, lit
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
        leafDict = row.asDict()
        leafDict['leaf'] = leaf
        return(Row(**leafDict))
    root = tree['dfs']
    data = self.rdd.map(lambda row: f_assign_leaf(row, root)).toDF()
    numSamples = min(data.groupby('leaf').count().toPandas()['count'].values)
    data = data.withColumn('row_num', row_number().over(Window.partitionBy('leaf').orderBy(lit(1))) - 1)
    data = data.filter('row_num < {}'.format(numSamples)).drop('row_num')
    return data

setattr(DataFrame, "treeLeafCells", treeLeafCells)

def representativeSamples(self, variables, J = [2, 4, 4], depth = None, batch_size = None, local_depth = None, seed = 0):
    from pyspark.sql import Window
    from pyspark.sql.functions import row_number, rand, lit
    from epstatKDTree import epstatKDTree, treeLeafCells
    tree = self.epstatKDTree(variables, J, depth, batch_size, local_depth)
    data = self.treeLeafCells(tree)
    data = data.withColumn('row_num', row_number().over(Window.partitionBy('leaf').orderBy(lit(1))) - 1).withColumn('rand', rand(seed=seed))
    data = data.withColumn('sample', row_number().over(Window.partitionBy('leaf').orderBy('rand')) - 1).drop('row_num', 'rand', 'leaf')
    return data

setattr(DataFrame, "representativeSamples", representativeSamples)

def randomSamples(self, variables, sample_size, seed = 0):
    from pyspark.sql import Window
    from pyspark.sql.functions import row_number, rand, lit
    numSamples = int(self.count()/sample_size)
    data = self.withColumn('index', row_number().over(Window.orderBy(lit(1)))%(lit(numSamples))).withColumn('rand', rand(seed=seed))
    data = data.withColumn('sample', row_number().over(Window.partitionBy('index').orderBy('rand')) - 1).drop('index', 'rand')
    return data

setattr(DataFrame, "randomSamples", randomSamples)
