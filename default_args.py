def method_default_args(prob_type):
    defaults = {}
    defaults['simpleVar'] = 100
    defaults['simpleIneq'] = 50
    defaults['simpleEq'] = 50
    defaults['simpleEx'] = 10000
    defaults['nonconvexVar'] = 100
    defaults['nonconvexIneq'] = 50
    defaults['nonconvexEq'] = 50
    defaults['nonconvexEx'] = 10000
    defaults['saveAllStats'] = True
    defaults['resultsSaveFreq'] = 50
    if 'im' in prob_type:
        defaults['epochs'] = 1000
        defaults['batchSize'] = 20
        defaults['lr'] = 1e-4
        defaults['hiddenSize'] = 200
        defaults['softWeight'] = 10       # use 100 if useCompl=False, 10 otherwise
        defaults['softWeightEqFrac'] = 1.0
        defaults['useCompl'] = True
        defaults['useTrainCorr'] = True
        defaults['useTestCorr'] = True
        defaults['corrMode'] = 'partial'    # use 'full' if useCompl=False, that is, take gradient steps for both eq and ineq
                                         # when we don't do completion to satisfy eq first
        defaults['corrTrainSteps'] = 10
        defaults['corrTestMaxSteps'] = 10
        defaults['corrEps'] = 1e-4
        defaults['corrLr'] = 1e-7
        defaults['corrMomentum'] = 0.5
    else:
        raise NotImplementedError

    return defaults