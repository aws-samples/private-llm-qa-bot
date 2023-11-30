import json
from typing import Any, Dict, List, Union,Mapping, Optional, TypeVar, Union

import boto3


def query_ec2_price(**args) -> Union[str,None]:  
    region = args.get('region','cn-northwest-1')
    term = args.get('term','OnDemand')
    instance_type = args.get('instance_type','m5.large')
    os = args.get('os','Linux')
    if not region.startswith('cn-'):
        pricing_client = boto3.client('pricing', region_name='us-east-1')
    else:
        pricing_client = boto3.client('pricing', region_name='cn-northwest-1')

    def parse_price(products,term):
        ret = []
        for product in products:
            product = json.loads(product)
            on_demand_terms = product['terms'].get(term)
            if on_demand_terms:
                for _, term_details in on_demand_terms.items():
                    price_dimensions = term_details['priceDimensions']
                    for _, price_dimension in price_dimensions.items():
                        price = price_dimension['pricePerUnit']['CNY'] if region.startswith('cn-') else price_dimension['pricePerUnit']['USD']
                        desc =  price_dimension['description']
                        if not desc.startswith("$0.00 per") and not desc.startswith("USD 0.0 per"):
                            ret.append(f"Region: {region}, Price per unit: {price}, description: {desc}")
        return ret
    
    response = pricing_client.get_products(
        ServiceCode='AmazonEC2',
        Filters=[
            {
                'Type': 'TERM_MATCH',
                'Field': 'instanceType',
                'Value': instance_type 
            },
            {
                'Type': 'TERM_MATCH',
                'Field': 'ServiceCode',
                'Value': 'AmazonEC2'
            },
            {
                'Type': 'TERM_MATCH',
                'Field': 'regionCode',
                'Value': region
            },
            {
                'Type': 'TERM_MATCH',
                'Field': 'tenancy',
                'Value': 'Shared'
            },
            {
                'Type': 'TERM_MATCH',
                'Field': 'operatingSystem',
                'Value': os
            },
        ]
    )
    products = response['PriceList']
    prices = parse_price(products,term=term)
    
    return '\n'.join(prices) if prices else None


if __name__ == "__main__":
    print(query_ec2_price(instance_type='m5.xlarge',region='us-east-1'))