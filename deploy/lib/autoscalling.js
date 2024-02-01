// import * as autoscaling from 'aws-cdk-lib/aws-autoscaling';
// import * as appscaling from 'aws-cdk-lib/aws-applicationautoscaling';
// import {v4 as uuidv4} from 'uuid';


export function addAutoScaling(fn,minCapacity=1,maxCapacity=100){
    const alias = fn.addAlias('prod');

    // Create AutoScaling target
    const as = alias.addAutoScaling({ minCapacity:minCapacity, maxCapacity: maxCapacity });
    
    // Configure Target Tracking
    as.scaleOnUtilization({
      utilizationTarget: 0.7,
      minCapacity: minCapacity,
      maxCapacity:maxCapacity,
    });
    return alias
}

export function addAutoScalingDDb(table,minCapacity=5,maxCapacity=1000){
    const readCapacity = table.autoScaleReadCapacity({
      minCapacity: minCapacity,
      maxCapacity: maxCapacity
    });
    readCapacity.scaleOnUtilization({
      targetUtilizationPercent: 60
    });
    const writeCapacity = table.autoScaleWriteCapacity({
      minCapacity: minCapacity,
      maxCapacity: maxCapacity
    });
    writeCapacity.scaleOnUtilization({
      targetUtilizationPercent: 60
    });
}
