import dsl

DSL_DICT = {('list', 'list') : [dsl.MapFunction, dsl.MapPrefixesFunction, dsl.SimpleITE],
                        ('list', 'atom') : [dsl.FoldFunction, dsl.running_averages.RunningAverageLast5Function, dsl.SimpleITE, 
                                            dsl.running_averages.RunningAverageLast10Function, dsl.running_averages.RunningAverageWindow11Function,
                                            dsl.running_averages.RunningAverageWindow5Function],
                        ('atom', 'atom') : [dsl.SimpleITE, dsl.AddFunction, dsl.MultiplyFunction, dsl.sk152.ArmSelection, 
                                            dsl.sk152.LegSelection, 
                                            dsl.sk152.FaceSelection]}


CUSTOM_EDGE_COSTS = {
    ('list', 'list') : {},
    ('list', 'atom') : {},
    ('atom', 'atom') : {}
}

