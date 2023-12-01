import json
from typing import Any, Dict, List, Union,Mapping, Optional, TypeVar, Union

import boto3


def purchase_option_filter(term_attri:dict, value:str) -> dict:
    ##如果是空值，则不判断
    if not value:
        return True
    if term_attri:
        purchaseOption = term_attri.get('PurchaseOption')
        if purchaseOption == value:
            return True
    return None


def query_ec2_price(**args) -> Union[str,None]:  
    region = args.get('region','cn-northwest-1')
    term = args.get('term','OnDemand')
    instance_type = args.get('instance_type','m5.large')
    os = args.get('os','Linux')
    purchase_option = args.get('purchase_option','')
    if not region.startswith('cn-'):
        pricing_client = boto3.client('pricing', region_name='us-east-1')
    else:
        pricing_client = boto3.client('pricing', region_name='cn-northwest-1')

    def parse_price(products,args):
        ret = []
        for product in products:
            product = json.loads(product)
            on_demand_terms = product['terms'].get(term)
            if on_demand_terms and term == 'Reserved':
                for _, term_details in on_demand_terms.items():
                    price_dimensions = term_details['priceDimensions']
                    term_attri = term_details.get('termAttributes')
                    is_valid = purchase_option_filter(term_attri,purchase_option)
                    option = term_attri.get('PurchaseOption')
                    if is_valid:
                        for _, price_dimension in price_dimensions.items():
                            price = price_dimension['pricePerUnit']['CNY'] if region.startswith('cn-') else price_dimension['pricePerUnit']['USD']
                            dollar = 'CNY' if region.startswith('cn-') else 'USD'
                            desc =  price_dimension['description']
                            unit =  price_dimension['unit']
                            if not desc.startswith("$0.00 per") and not desc.startswith("USD 0.0 per") \
                                    and not desc.startswith("0.00 CNY per") and not desc.startswith("CNY 0.0 per"):
                                ret.append(f"Purchase option: {option}, Lease contract length: {term_attri.get('LeaseContractLength')}, Offering Class: {term_attri.get('OfferingClass')}, Price per {unit}: {dollar} {price}, description: {desc}")
            elif on_demand_terms:
                for _, term_details in on_demand_terms.items():
                    price_dimensions = term_details['priceDimensions']
                    if price_dimensions:
                        for _, price_dimension in price_dimensions.items():
                            price = price_dimension['pricePerUnit']['CNY'] if region.startswith('cn-') else price_dimension['pricePerUnit']['USD']
                            desc =  price_dimension['description']
                            unit =  price_dimension['unit']
                            if not desc.startswith("$0.00 per") and not desc.startswith("USD 0.0 per") and not desc.startswith("0.00 CNY per"):
                                ret.append(f"Region: {region}, Price per {unit}: {price}, description: {desc}")
        return ret
    
    filters = [
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
            }
        ]
    
    if purchase_option:
        filters = filters + [{
                    'Type': 'TERM_MATCH',
                    'Field': 'PurchaseOption',
                    'Value': purchase_option
                }] 
        
    response = pricing_client.get_products(
        ServiceCode='AmazonEC2',
        Filters=filters
    )
    products = response['PriceList']
    prices = parse_price(products,args)
    
    return '\n'.join(prices) if prices else None


if __name__ == "__main__":
    print(query_ec2_price(instance_type='m5.xlarge',region='cn-northwest-1',term='OnDemand',purchase_option=''))