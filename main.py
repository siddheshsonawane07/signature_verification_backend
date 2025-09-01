"""
Test Script for Signature Verification System
Run this to test enrollment and verification
"""

import sys
from pathlib import Path
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ml.verification.signature_verifier import SignatureVerifier

def test_system():
    """Test the complete verification system"""
    
    print("="*50)
    print("TESTING SIGNATURE VERIFICATION SYSTEM")
    print("="*50)
    
    # Initialize verifier
    verifier = SignatureVerifier()
    verifier.load_or_create_model()
    
    # Check for enrolled users
    enrolled_users = verifier.get_enrolled_users()
    print(f"Currently enrolled users: {enrolled_users}")
    
    if not enrolled_users:
        print("\nNo users enrolled yet. Let's enroll some users...")
        enroll_users_from_data(verifier)
    else:
        print("\nTesting verification with existing users...")
        test_verification_with_existing_users(verifier)

def enroll_users_from_data(verifier):
    """Enroll users from data directory"""
    
    users_dir = Path("data/users")
    if not users_dir.exists():
        print("Error: data/users directory not found!")
        return
    
    enrolled_count = 0
    
    for user_folder in users_dir.iterdir():
        if user_folder.is_dir():
            train_dir = user_folder / "train"
            if train_dir.exists():
                # Get signature files
                signature_files = []
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    signature_files.extend(train_dir.glob(ext))
                
                if len(signature_files) >= 3:
                    try:
                        # Enroll user
                        signature_paths = [str(f) for f in signature_files[:8]]  # Max 8 signatures
                        verifier.enroll_user(user_folder.name, signature_paths)
                        enrolled_count += 1
                        print()  # Empty line for readability
                    except Exception as e:
                        print(f"Failed to enroll {user_folder.name}: {e}")
                else:
                    print(f"Skipped {user_folder.name}: only {len(signature_files)} signatures (need 3+)")
    
    print(f"\nEnrolled {enrolled_count} users successfully!")
    
    if enrolled_count > 0:
        print("\nNow testing verification...")
        test_verification_with_existing_users(verifier)

def test_verification_with_existing_users(verifier):
    """Test verification with enrolled users"""
    
    enrolled_users = verifier.get_enrolled_users()
    
    for username in enrolled_users[:2]:  # Test first 2 users
        print(f"\nTesting verification for user: {username}")
        
        # Get user info
        user_info = verifier.get_user_info(username)
        print(f"  Enrolled: {user_info['enrollment_date']}")
        print(f"  Signatures: {user_info['signature_count']}")
        print(f"  Threshold: {user_info['threshold']:.3f}")
        
        # Test with training data (should verify as genuine)
        train_dir = Path(f"data/users/{username}/train")
        test_dir = Path(f"data/users/{username}/test")
        
        # Test with a training signature (should pass)
        train_files = list(train_dir.glob("*.png")) + list(train_dir.glob("*.jpg"))
        if train_files:
            print(f"\n  Testing with training signature: {train_files[0].name}")
            result = verifier.verify_signature(username, str(train_files[0]))
            print(f"  Result: {'PASS' if result['verified'] else 'FAIL'}")
        
        # Test with test signatures if available
        if test_dir.exists():
            test_files = list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg"))
            if test_files:
                print(f"\n  Testing with test signature: {test_files[0].name}")
                result = verifier.verify_signature(username, str(test_files[0]))
                print(f"  Result: {'PASS' if result['verified'] else 'FAIL'}")

def interactive_test():
    """Interactive testing mode"""
    
    verifier = SignatureVerifier()
    verifier.load_or_create_model()
    
    while True:
        print("\n" + "="*40)
        print("INTERACTIVE SIGNATURE VERIFICATION")
        print("="*40)
        print("1. Enroll new user")
        print("2. Verify signature") 
        print("3. List enrolled users")
        print("4. Auto-enroll from data directory")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            username = input("Enter username: ").strip()
            signature_dir = input("Enter path to signature directory: ").strip()
            
            signature_dir_path = Path(signature_dir)
            if signature_dir_path.exists():
                signature_files = []
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    signature_files.extend(signature_dir_path.glob(ext))
                
                if len(signature_files) >= 3:
                    try:
                        signature_paths = [str(f) for f in signature_files]
                        verifier.enroll_user(username, signature_paths)
                    except Exception as e:
                        print(f"Enrollment failed: {e}")
                else:
                    print(f"Need at least 3 signatures, found {len(signature_files)}")
            else:
                print("Directory not found!")
        
        elif choice == "2":
            username = input("Enter username: ").strip()
            signature_path = input("Enter path to test signature: ").strip()
            
            if Path(signature_path).exists():
                result = verifier.verify_signature(username, signature_path)
                if 'error' in result:
                    print(f"Error: {result['error']}")
                else:
                    print(f"\nVerification Result:")
                    print(f"  Verified: {result['verified']}")
                    print(f"  Confidence: {result['confidence']:.1f}%")
                    print(f"  Max Similarity: {result['max_similarity']:.3f}")
                    print(f"  Threshold: {result['threshold']:.3f}")
            else:
                print("Signature file not found!")
        
        elif choice == "3":
            users = verifier.get_enrolled_users()
            if users:
                print(f"\nEnrolled users ({len(users)}):")
                for user in users:
                    info = verifier.get_user_info(user)
                    print(f"  - {user}: {info['signature_count']} signatures, enrolled {info['enrollment_date']}")
            else:
                print("No users enrolled yet")
        
        elif choice == "4":
            print("Auto-enrolling users from data directory...")
            enroll_users_from_data(verifier)
        
        elif choice == "5":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice!")

def quick_test():
    """Quick test with sample data"""
    print("="*50)
    print("QUICK TEST - SIGNATURE VERIFICATION")
    print("="*50)
    
    # Check if model exists
    if not os.path.exists("data/models/siamese_model.h5"):
        print("Error: No trained model found!")
        print("Please run training first: python ml/training/train_siamese.py")
        return
    
    # Check if data exists
    users_dir = Path("data/users")
    if not users_dir.exists() or not any(users_dir.iterdir()):
        print("Error: No user data found!")
        print("Please add signature images to data/users/{username}/train/ folders")
        return
    
    # Run automatic test
    test_system()

def verify_single_signature():
    """Verify a single signature (command line utility)"""
    
    if len(sys.argv) != 3:
        print("Usage: python test_verification.py --verify <username> <signature_path>")
        return
    
    username = sys.argv[1]
    signature_path = sys.argv[2]
    
    verifier = SignatureVerifier()
    verifier.load_or_create_model()
    
    result = verifier.verify_signature(username, signature_path)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"User: {username}")
        print(f"Signature: {signature_path}")
        print(f"Verified: {result['verified']}")
        print(f"Confidence: {result['confidence']:.1f}%")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Signature Verification System')
    parser.add_argument('--mode', choices=['auto', 'interactive', 'quick'], default='quick',
                       help='Test mode: auto, interactive, or quick')
    parser.add_argument('--verify', nargs=2, metavar=('USERNAME', 'SIGNATURE_PATH'),
                       help='Verify single signature: --verify username signature_path')
    
    args = parser.parse_args()
    
    if args.verify:
        sys.argv = ['test_verification.py'] + args.verify
        verify_single_signature()
    elif args.mode == 'interactive':
        interactive_test()
    elif args.mode == 'auto':
        test_system()
    else:  # quick mode
        quick_test()