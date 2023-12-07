export function addAutoScaling(fn,minCapacity=1,maxCapacity=100){
  const alias = fn.addAlias('prod');

  // Create AutoScaling target
  const as = alias.addAutoScaling({ minCapacity:minCapacity, maxCapacity: maxCapacity });
  
  // Configure Target Tracking
  as.scaleOnUtilization({
    utilizationTarget: 0.5,
    minCapacity: minCapacity,
    maxCapacity:maxCapacity,
  });
  return alias
}