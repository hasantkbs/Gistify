import 'package:flutter/material.dart';

import '../../theme/app_colors.dart';
import '../../widgets/option_tile.dart';

class UploadPage extends StatelessWidget {
  const UploadPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  IconButton(
                    onPressed: () => Navigator.maybePop(context),
                    icon: const Icon(Icons.close, size: 26),
                  ),
                  const Spacer(),
                  const Text(
                    'Upload',
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.w700),
                  ),
                  const Spacer(),
                  const SizedBox(width: 48),
                ],
              ),
              const SizedBox(height: 32),
              const Text(
                'Choose a file',
                style: TextStyle(
                  fontSize: 28,
                  fontWeight: FontWeight.w700,
                  letterSpacing: -0.3,
                ),
              ),
              const SizedBox(height: 28),
              const OptionTile(
                icon: Icons.smartphone,
                title: 'From device',
                trailing: Icon(Icons.chevron_right),
                backgroundColor: AppColors.tileBackground,
              ),
              const OptionTile(
                icon: Icons.cloud_outlined,
                title: 'From cloud',
                trailing: Icon(Icons.chevron_right),
                backgroundColor: AppColors.tileBackground,
              ),
              const OptionTile(
                icon: Icons.photo_camera_outlined,
                title: 'Take a photo',
                trailing: Icon(Icons.chevron_right),
                backgroundColor: AppColors.tileBackground,
              ),
              const SizedBox(height: 32),
              Text(
                'Add notes or tags',
                style: Theme.of(
                  context,
                ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600),
              ),
              const SizedBox(height: 12),
              TextFormField(
                maxLines: 4,
                decoration: const InputDecoration(
                  hintText: 'Add your notes here...',
                ),
              ),
              const Spacer(),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(
                  horizontal: 20,
                  vertical: 18,
                ),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(24),
                  boxShadow: const [
                    BoxShadow(
                      color: Color(0x11000000),
                      offset: Offset(0, -2),
                      blurRadius: 20,
                    ),
                  ],
                ),
                child: Row(
                  children: const [
                    Text(
                      'Uploading...',
                      style: TextStyle(fontWeight: FontWeight.w600),
                    ),
                    Spacer(),
                    Text(
                      '50%',
                      style: TextStyle(
                        fontWeight: FontWeight.w700,
                        color: AppColors.primaryBlue,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
