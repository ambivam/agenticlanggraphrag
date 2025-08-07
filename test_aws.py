import os
import boto3
from dotenv import load_dotenv

def test_aws_credentials():
    print("Testing AWS Credentials...")
    
    # Load environment variables
    load_dotenv()
    
    # Get credentials
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region = os.getenv('AWS_REGION', 'us-east-1')
    
    print(f"\nChecking environment variables:")
    print(f"AWS_ACCESS_KEY_ID: {'✓ Set' if access_key else '✗ Not set'}")
    print(f"AWS_SECRET_ACCESS_KEY: {'✓ Set' if secret_key else '✗ Not set'}")
    print(f"AWS_REGION: {region}")
    
    try:
        # Create S3 client
        print("\nTrying to connect to AWS...")
        s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        
        # Try to list buckets
        print("Listing S3 buckets...")
        response = s3_client.list_buckets()
        buckets = [bucket['Name'] for bucket in response['Buckets']]
        print(f"\n✓ Success! Found {len(buckets)} buckets:")
        for bucket in buckets:
            print(f"  - {bucket}")
            
        # Try to list files in specific bucket
        if 'brkbluespirebucket' in buckets:
            print(f"\nListing files in brkbluespirebucket...")
            response = s3_client.list_objects_v2(Bucket='brkbluespirebucket')
            if 'Contents' in response:
                files = [item['Key'] for item in response['Contents']]
                print(f"Found {len(files)} files:")
                for file in files[:5]:  # Show first 5 files
                    print(f"  - {file}")
            else:
                print("No files found in bucket")
                
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        if 'InvalidAccessKeyId' in str(e):
            print("\nThe AWS_ACCESS_KEY_ID is invalid. Please check:")
            print("1. The key is copied correctly with no extra spaces")
            print("2. The key exists in your AWS IAM console")
        elif 'SignatureDoesNotMatch' in str(e):
            print("\nThe AWS_SECRET_ACCESS_KEY is invalid. Please check:")
            print("1. The key is copied correctly with no extra spaces")
            print("2. The key matches the access key ID in AWS")
        elif 'NoSuchBucket' in str(e):
            print("\nThe bucket does not exist or you don't have access to it")
        else:
            print("\nPlease check:")
            print("1. Your AWS credentials are correct")
            print("2. The region matches your bucket's region")
            print("3. You have the necessary S3 permissions")

if __name__ == "__main__":
    test_aws_credentials()
