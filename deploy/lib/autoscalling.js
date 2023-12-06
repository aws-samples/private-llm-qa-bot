import * as autoscaling from 'aws-cdk-lib/aws-autoscaling';

export function addAutoScaling(fn,minCapacity=3,maxCapacity=100){
    const alias = fn.addAlias('prod');

    // Create AutoScaling target
    const as = alias.addAutoScaling({ minCapacity:minCapacity, maxCapacity: maxCapacity });
    
    // Configure Target Tracking
    as.scaleOnUtilization({
      utilizationTarget: 0.5,
      minCapacity: minCapacity,
    });
    
    // // Configure Scheduled Scaling
    // as.scaleOnSchedule('ScaleUpInTheMorning', {
    //   schedule: autoscaling.Schedule.cron({ hour: '8', minute: '0'}),
    //   minCapacity: 10,
    // });

}
