from investigations.accuracy import AccuracyInvestigation

def main():
    print("=" * 50)
    print("ARGUS VISION")
    print("=" * 50)

    investigation = AccuracyInvestigation()
    investigation.run(None, None)

if __name__ == "__main__":
    main()