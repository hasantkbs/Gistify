import 'package:flutter/material.dart';

import '../../theme/app_colors.dart';

class DataStoragePage extends StatefulWidget {
  const DataStoragePage({super.key});

  static const route = '/data-storage';

  @override
  State<DataStoragePage> createState() => _DataStoragePageState();
}

class _DataStoragePageState extends State<DataStoragePage> {
  bool syncEnabled = true;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(24, 16, 24, 32),
          children: [
            _pageHeader(context, 'Data & Storage'),
            const SizedBox(height: 24),
            const Text(
              'Storage',
              style: TextStyle(fontWeight: FontWeight.w700, fontSize: 18),
            ),
            const SizedBox(height: 16),
            _StorageTile(
              icon: Icons.archive_outlined,
              title: 'Current Storage',
              subtitle: '1.2 GB used',
              color: AppColors.primaryBlue,
            ),
            _StorageTile(
              icon: Icons.delete_outline,
              title: 'Clear Cache',
              subtitle: 'Free up space',
              trailing: const Text(
                '100 MB',
                style: TextStyle(
                  color: Colors.redAccent,
                  fontWeight: FontWeight.w700,
                ),
              ),
              color: Colors.redAccent,
            ),
            _StorageTile(
              icon: Icons.download_done_outlined,
              title: 'Downloaded Summaries',
              subtitle: 'Manage offline files',
              trailing: const Text(
                '200 MB',
                style: TextStyle(fontWeight: FontWeight.w700),
              ),
              color: AppColors.primaryBlue,
            ),
            const SizedBox(height: 28),
            const Text(
              'Data',
              style: TextStyle(fontWeight: FontWeight.w700, fontSize: 18),
            ),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(20),
              ),
              child: Row(
                children: [
                  Container(
                    width: 48,
                    height: 48,
                    decoration: BoxDecoration(
                      color: Colors.greenAccent.withValues(alpha: 0.2),
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: const Icon(Icons.sync, color: AppColors.primaryBlue),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: const [
                        Text(
                          'Data Synchronization',
                          style: TextStyle(
                            fontWeight: FontWeight.w700,
                            fontSize: 16,
                          ),
                        ),
                        SizedBox(height: 4),
                        Text(
                          'Sync data across devices',
                          style: TextStyle(color: AppColors.textSecondary),
                        ),
                      ],
                    ),
                  ),
                  Switch.adaptive(
                    value: syncEnabled,
                    onChanged: (value) {
                      setState(() => syncEnabled = value);
                    },
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _pageHeader(BuildContext context, String title) {
    return Row(
      children: [
        IconButton(
          onPressed: () => Navigator.pop(context),
          icon: const Icon(Icons.arrow_back_ios_new),
        ),
        const Spacer(),
        Text(
          title,
          style: const TextStyle(fontSize: 24, fontWeight: FontWeight.w700),
        ),
        const Spacer(),
        const SizedBox(width: 48),
      ],
    );
  }
}

class _StorageTile extends StatelessWidget {
  const _StorageTile({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.color,
    this.trailing,
  });

  final IconData icon;
  final String title;
  final String subtitle;
  final Widget? trailing;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          children: [
            Container(
              width: 48,
              height: 48,
              decoration: BoxDecoration(
                color: color.withValues(alpha: 0.18),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Icon(icon, color: color),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: const TextStyle(
                      fontWeight: FontWeight.w700,
                      fontSize: 16,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    subtitle,
                    style: const TextStyle(color: AppColors.textSecondary),
                  ),
                ],
              ),
            ),
            if (trailing != null) trailing!,
          ],
        ),
      ),
    );
  }
}
